from datetime import datetime
from email.utils import format_datetime

import pandas as pd
import pytest
from conftest import FIXTURES, needs_injury_pdfs

from nba_sos.ingestion import injury_reports as ir
from nba_sos.ingestion.raw_store import RawStore
from nba_sos.preprocessing.availability import nickname_keys

NEW_START = datetime(2025, 12, 22, 9, 0)
PDFS = FIXTURES / "injury_reports"


# ------------------------------------------------------------------ slots / names
def test_legacy_slots_are_hourly_and_published_at_half_past():
    tip = datetime(2023, 11, 10, 19, 0)
    slots = ir.slot_nominal_times(tip, NEW_START, 3)
    assert slots == [datetime(2023, 11, 10, 18, 30), datetime(2023, 11, 10, 17, 30), datetime(2023, 11, 10, 16, 30)]
    assert ir.slot_filename(slots[0], NEW_START) == "Injury-Report_2023-11-10_06PM.pdf"
    # 19:30 tip-off: the 07PM file (published 19:30) is NOT strictly earlier
    assert ir.slot_nominal_times(datetime(2023, 11, 10, 19, 30), NEW_START, 1) == [datetime(2023, 11, 10, 18, 30)]
    assert ir.slot_nominal_times(datetime(2023, 11, 10, 19, 45), NEW_START, 1) == [datetime(2023, 11, 10, 19, 30)]


def test_new_format_slots_are_quarter_hourly():
    tip = datetime(2026, 1, 10, 19, 0)
    slots = ir.slot_nominal_times(tip, NEW_START, 2)
    assert slots == [datetime(2026, 1, 10, 18, 45), datetime(2026, 1, 10, 18, 30)]
    assert ir.slot_filename(slots[0], NEW_START) == "Injury-Report_2026-01-10_06_45PM.pdf"


@needs_injury_pdfs
# ------------------------------------------------------------------ parsing (real PDFs)
@pytest.mark.parametrize(
    "name,players,not_submitted,pages",
    [
        ("2019-11-15_05PM.pdf", 132, 7, 6),  # Nov-Dec 2019: Reason before Current Status
        ("2019-12-08_05PM.pdf", 111, 11, 6),  # 2019-20 variant with a Previous Status column
        ("2021-12-01_05PM.pdf", 102, 6, 5),  # spaced text, header on every page
        ("2023-11-10_05PM.pdf", 111, 2, 7),  # unspaced text, header on page 1 only
        ("2026-11-17_05_30PM.pdf", 123, 10, 9),  # current format
    ],
)
def test_parse_real_reports(name, players, not_submitted, pages):
    df, info = ir.parse_report_pdf(PDFS / name)
    r = ir.resolve_rows(df, nickname_keys())
    assert info["pages"] == pages and info["orphan_fragments"] == 0
    assert int((~r.not_submitted).sum()) == players
    assert int(r.not_submitted.sum()) == not_submitted
    assert r.status_valid.all()
    assert r.team_tricode.notna().all()
    assert r.away_tricode.notna().all() and r.game_date.notna().all()


@needs_injury_pdfs
def test_parse_2021_matches_pdfplumber_tables_and_header_time():
    import pdfplumber

    df, info = ir.parse_report_pdf(PDFS / "2021-12-01_05PM.pdf")
    assert info["report_time_et"] == datetime(2021, 12, 1, 17, 30)
    with pdfplumber.open(PDFS / "2021-12-01_05PM.pdf") as p:
        table_players = [
            (row[4], row[5]) for pg in p.pages for t in pg.extract_tables() for row in t[1:] if row[4] and row[5]
        ]
    parsed = list(zip(df[~df.not_submitted].player, df[~df.not_submitted].status))
    assert parsed == table_players


@needs_injury_pdfs
def test_wrapped_reason_is_reattached():
    df, _ = ir.parse_report_pdf(PDFS / "2026-11-17_05_30PM.pdf")
    row = df[df.player == "Bittle,Nate"].iloc[0]
    assert row.reason == "Injury/Illness-RightFootAvulsion Fracture"
    assert row.matchup == "NYK@TOR" and row.status == "Out"


def test_team_resolution_prefers_longest_nickname():
    # "charlottehornets" ends with both "hornets" and "nets"
    df = pd.DataFrame(
        {
            "game_date": ["10/24/2021"],
            "matchup": ["CHA@BKN"],
            "team": ["Charlotte Hornets"],
            "player": ["Rozier, Terry"],
            "status": ["Out"],
            "not_submitted": [False],
        }
    )
    assert ir.resolve_rows(df, nickname_keys()).team_tricode.iloc[0] == "CHA"


@pytest.mark.parametrize(
    "raw,key",
    [
        ("Porter Jr., Michael", "michaelporterjr"),
        ("PorterJr.,Michael", "michaelporterjr"),
        ("Bogdanović, Bogdan", "bogdanbogdanovic"),
        ("Wright IV, McKinley", "mckinleywrightiv"),
    ],
)
def test_player_key_normalisation(raw, key):
    assert ir.player_key_from_report(raw) == key


# ------------------------------------------------------------------ fetch walk-back
def test_fetch_walks_back_and_requires_publication_before_tipoff(tmp_path):
    cfg = {
        "base_url": "https://x/Injury-Report_",
        "new_start_et": "2025-12-22T09:00",
        "max_slot_lookback": 4,
        "sleep_between_requests_s": 0,
        "request_timeout_s": 5,
        "user_agent": "t",
    }
    tip = pd.Timestamp("2023-11-11 00:00", tz="UTC")  # 19:00 ET
    pdf = b"%PDF-1.4 fake"
    served = {
        # 06PM exists but was (re)uploaded AFTER tip-off -> must be rejected
        "https://x/Injury-Report_2023-11-10_06PM.pdf": (200, pdf, pd.Timestamp("2023-11-11 01:00", tz="UTC")),
        # 05PM missing
        "https://x/Injury-Report_2023-11-10_05PM.pdf": (404, b"", None),
        "https://x/Injury-Report_2023-11-10_04PM.pdf": (200, pdf, pd.Timestamp("2023-11-10 21:30", tz="UTC")),
    }
    calls = []

    def client(url, timeout, ua):
        calls.append(url)
        s, body, lm = served.get(url, (404, b"", None))
        return s, body, format_datetime(lm.to_pydatetime(), usegmt=True) if lm is not None else None

    st = RawStore(tmp_path)
    games = pd.DataFrame({"game_id": ["g1"], "tipoff_utc": [tip]})
    res = ir.fetch_for_games(st, games, cfg, client=client, sleep=lambda s: None)
    assert res.report_path.iloc[0] == "injury_reports/2023-11-10/Injury-Report_2023-11-10_04PM.pdf"
    # 404 is remembered: a second run makes no request for it
    calls.clear()
    ir.fetch_for_games(st, games, cfg, client=client, sleep=lambda s: None)
    assert calls == []


@needs_injury_pdfs
def test_parse_2018_19_nine_column_layout():
    """2018-19 reports carry Category and Previous Status columns."""
    df, info = ir.parse_report_pdf(PDFS / "2019-04-13_05PM.pdf")
    r = ir.resolve_rows(df, nickname_keys())
    assert info["report_time_et"] == datetime(2019, 4, 13, 17, 30)
    assert int((~r.not_submitted).sum()) == 23 and int(r.not_submitted.sum()) == 3
    assert r.status_valid.all() and r.team_tricode.notna().all()
    embiid = r[r.player == "Embiid, Joel"].iloc[0]
    assert embiid.status == "Available" and embiid.team_tricode == "PHI"


def test_network_errors_are_retried_then_skipped_not_fatal(tmp_path):
    import requests as rq

    cfg = {
        "base_url": "https://x/Injury-Report_",
        "sleep_between_requests_s": 0,
        "request_timeout_s": 5,
        "user_agent": "t",
    }
    calls = []

    def flaky(url, timeout, ua):
        calls.append(url)
        raise rq.exceptions.ConnectionError("dns")

    rec = ir.fetch_report(
        RawStore(tmp_path), "Injury-Report_2020-08-06_02PM.pdf", cfg, {}, client=flaky, sleep=lambda s: None
    )
    assert rec["path"] is None and rec["http_status"] is None and len(calls) == ir.NETWORK_RETRIES
    assert RawStore(tmp_path).read_manifest() == []  # not recorded as missing: retried next run
