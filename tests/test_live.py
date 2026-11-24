"""Live flow without the network: report slot walk-back, written outputs, and the Streamlit page."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from nba_sos import live
from nba_sos.config import PROJECT_ROOT, Paths, load_configs

APP = PROJECT_ROOT / "app" / "streamlit_app.py"


@pytest.fixture
def tmp_paths(tmp_path):
    class TmpPaths(Paths):
        raw = property(lambda self: tmp_path / "raw")
        reports = property(lambda self: tmp_path / "reports")

    return TmpPaths(PROJECT_ROOT)


def test_latest_report_skips_recent_slots_and_walks_back(tmp_paths):
    cfg = load_configs(Paths(PROJECT_ROOT))
    asked = []

    def client(url, timeout, user_agent):
        asked.append(url.rsplit("_", 3)[-3:])
        if len(asked) < 3:
            return 403, b"", None
        return 200, b"%PDF-1.4 test", "Tue, 24 Nov 2026 18:46:00 GMT"

    now = datetime(2026, 11, 24, 20, 0, tzinfo=UTC)  # 15:00 ET
    rec = live.latest_injury_report(tmp_paths, cfg, now_utc=now, client=client, sleep=lambda s: None)
    # slots strictly before 14:30 ET (30 minutes before now), newest first
    assert asked == [
        ["2026-11-24", "02", "15PM.pdf"],
        ["2026-11-24", "02", "00PM.pdf"],
        ["2026-11-24", "01", "45PM.pdf"],
    ]
    assert rec["path"] == "injury_reports/2026-11-24/Injury-Report_2026-11-24_01_45PM.pdf"
    assert rec["published_at_utc"] == "2026-11-24T18:46:00+00:00"
    # known files and recorded absences are not requested again
    again = live.latest_injury_report(tmp_paths, cfg, now_utc=now, client=client, sleep=lambda s: None)
    assert again["path"] == rec["path"] and len(asked) == 3


def _fake_projection(snapshot: str) -> dict:
    table = pd.DataFrame(
        {
            "team_id": [1, 2],
            "team": ["AAA", "BBB"],
            "conference": ["East", "West"],
            "wins_now": [10, 5],
            "games_now": [15, 15],
            "exp_wins": [50.2, 30.1],
            "p05": [42.0, 22.0],
            "p95": [58.0, 38.0],
            "p_playoffs": [0.9, 0.2],
            "p_top6": [0.7, 0.05],
            "exp_seed": [3.1, 11.0],
            "d_sos": [0.4, -0.4],
            "d_sos_rank": [1, 2],
            "S": [3.0, -3.0],
            "sd": [1.5, 1.6],
            "n_injured_out": [1, 0],
        }
    )
    meta = {
        "season": "2026-27",
        "snapshot": snapshot,
        "cutoff": "2026-11-24",
        "completed_games": 230,
        "remaining_scheduled_games": 1000,
        "injury_report_in_projection": None,
        "n_paths": 100,
    }
    return {"table": table, "meta": meta}


def test_run_live_writes_outputs(tmp_paths, monkeypatch):
    from nba_sos.ingestion import fetch_all
    from nba_sos.models import season_simulation as SIM

    stamp = "20261124T080000Z"
    monkeypatch.setattr(fetch_all, "snapshot_current_season", lambda p, c: p.raw / "current" / stamp)
    monkeypatch.setattr(live, "latest_injury_report", lambda p, c: None)
    seen = {}

    def fake_run_current(p, c, n_paths, report, report_rec):
        seen.update(report=report, report_rec=report_rec)
        return _fake_projection(stamp)

    monkeypatch.setattr(SIM, "run_current", fake_run_current)
    out = live.run_live(tmp_paths, {}, n_paths=100)
    d = tmp_paths.reports / "live" / stamp
    assert out["dir"] == d
    assert pd.read_csv(d / "projection.csv").team.tolist() == ["AAA", "BBB"]
    assert not (d / "injury_report.csv").exists()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["injury_report"] is None and meta["injury_report_used_in_projection"] is False
    assert meta["snapshot"] == stamp
    assert seen == {"report": None, "report_rec": None}  # no report found: none passed on


def test_streamlit_page_fetches_once_per_session(monkeypatch):
    from streamlit.testing.v1 import AppTest

    calls = []

    def fake_run_live(paths, cfg):
        calls.append(1)
        proj = _fake_projection("x")
        meta = {**proj["meta"], "injury_report": None}
        return {"dir": "reports/live/x", "table": proj["table"], "injury_report": None, "meta": meta}

    monkeypatch.setattr(live, "run_live", fake_run_live)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert len(calls) == 1
    at.selectbox[0].select("BBB").run()
    assert not at.exception and len(calls) == 1  # choosing a team does not fetch again
    assert at.subheader[0].value == "BBB"
    at.button[0].click().run()
    assert len(calls) == 2
