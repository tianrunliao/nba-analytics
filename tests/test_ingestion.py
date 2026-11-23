import json

import pandas as pd
import pytest
from conftest import needs_raw_subset

from nba_sos.ingestion import nba_stats
from nba_sos.ingestion.raw_store import RawStore

CFG = {"season_type": "Regular Season", "request_timeout_s": 5, "sleep_between_requests_s": 0,
       "max_retries": 3, "retry_backoff_s": 0}


def lgl_payload(n=2):
    headers = ["SEASON_ID", "TEAM_ID", "GAME_ID", "PTS"]
    return json.dumps({"resultSets": [{"name": "LeagueGameLog", "headers": headers,
                                       "rowSet": [["22023", 1, f"00223000{i:02d}", 100 + i] for i in range(n)]}]})


def test_raw_store_writes_payload_and_manifest(tmp_path):
    st = RawStore(tmp_path)
    rec = st.write("a/b.json", b"{}", {"source": "x"})
    assert (tmp_path / "a" / "b.json").read_bytes() == b"{}"
    assert rec["sha256"] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert st.read_manifest()[0]["path"] == "a/b.json" and "fetched_at" in rec


def test_fetch_stores_raw_text_and_skips_when_cached(tmp_path):
    st, calls = RawStore(tmp_path), []

    def client(name, season, stype, timeout):
        calls.append(name)
        return lgl_payload(), "http://example/leaguegamelog"

    rec = nba_stats.fetch(st, "league_game_log", "2023-24", CFG, client=client, sleep=lambda s: None)
    assert rec["endpoint"] == "leaguegamelog" and rec["season"] == "2023-24"
    df = nba_stats.load(st, "league_game_log", "2023-24")
    assert len(df) == 2 and list(df.columns) == ["SEASON_ID", "TEAM_ID", "GAME_ID", "PTS"]
    assert nba_stats.fetch(st, "league_game_log", "2023-24", CFG, client=client, sleep=lambda s: None) is None
    assert calls == ["league_game_log"]


def test_fetch_retries_then_succeeds(tmp_path):
    attempts = []

    def flaky(name, season, stype, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("boom")
        return lgl_payload(), "u"

    rec = nba_stats.fetch(RawStore(tmp_path), "league_game_log", "2023-24", CFG, client=flaky, sleep=lambda s: None)
    assert rec["attempt"] == 3


def test_fetch_raises_after_retries_and_never_stores_unparseable(tmp_path):
    st = RawStore(tmp_path)
    with pytest.raises(RuntimeError):
        nba_stats.fetch(st, "league_game_log", "2023-24", CFG,
                        client=lambda *a: ("<html>blocked</html>", "u"), sleep=lambda s: None)
    assert not st.exists(nba_stats.raw_relpath("league_game_log", "2023-24"))
    assert st.read_manifest() == []


def test_load_missing_payload_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        nba_stats.load(RawStore(tmp_path), "league_game_log", "1999-00")


@needs_raw_subset
def test_schedule_fixture_parses_with_nba_api_parser():
    from conftest import FIXTURES
    st = RawStore(FIXTURES / "raw_subset")
    df = nba_stats.load(st, "schedule", "2023-24")
    assert {"gameId", "gameDateTimeUTC", "arenaName", "isNeutral", "homeTeam_teamId"} <= set(df.columns)
    assert isinstance(df, pd.DataFrame) and len(df) == 6
