import pandas as pd
from conftest import make_games

from nba_sos.ingestion.raw_store import RawStore
from nba_sos.preprocessing import availability as av

CFG = {"new_start_et": "2025-12-22T09:00", "max_slot_lookback": 4}


def _manifest(store, name, published):
    store.append_manifest({"path": f"injury_reports/2022-11-01/{name}", "sha256": name,
                           "published_at_utc": published, "url": "u"})


def test_select_report_latest_before_tipoff(tmp_path):
    st = RawStore(tmp_path)
    # tip-off 19:00 ET = 23:00 UTC on 2022-11-01 (EDT)
    _manifest(st, "Injury-Report_2022-11-01_06PM.pdf", "2022-11-01T23:05:00+00:00")  # after tip-off
    _manifest(st, "Injury-Report_2022-11-01_05PM.pdf", "2022-11-01T21:30:00+00:00")
    _manifest(st, "Injury-Report_2022-11-01_04PM.pdf", "2022-11-01T20:30:00+00:00")
    g = make_games([("2022-23", "2022-11-01", "AAA", "BBB", 100, 90)])
    sel = av.select_reports(g, st, CFG)
    assert sel.report_path.iloc[0].endswith("_05PM.pdf")
    assert sel.report_published_at_utc.iloc[0] < g.tipoff_utc.iloc[0]


def test_select_report_none_when_nothing_published_before_tipoff(tmp_path):
    st = RawStore(tmp_path)
    _manifest(st, "Injury-Report_2022-11-01_06PM.pdf", "2022-11-01T23:05:00+00:00")
    g = make_games([("2022-23", "2022-11-01", "AAA", "BBB", 100, 90)])
    assert av.select_reports(g, st, CFG).report_path.isna().all()


def test_build_availability_matching_and_team_status():
    g = make_games([("2022-23", "2022-11-01", "AAA", "BBB", 100, 90),
                    ("2022-23", "2022-11-01", "CCC", "DDD", 100, 90)])
    sel = pd.DataFrame({"game_id": g.game_id, "report_path": ["r.pdf", "r.pdf"],
                        "report_published_at_utc": pd.to_datetime(["2022-11-01 21:30"] * 2, utc=True),
                        "report_sha256": "x"})
    d = pd.Timestamp("2022-11-01").date()
    parsed = pd.DataFrame({
        "report_path": "r.pdf", "game_date": [d, d, d, d],
        "away_tricode": ["BBB", "BBB", "DDD", "BBB"], "home_tricode": ["AAA", "AAA", "CCC", "AAA"],
        "team_tricode": ["AAA", "BBB", "DDD", "AAA"],
        "player": ["One, P", "Two, P", None, "Gone, P"], "player_key": ["pone", "ptwo", None, "pgone"],
        "status": ["Out", "Questionable", None, "Out"], "reason": "", "not_submitted": [False, False, True, False],
        "status_valid": True, "game_time": "", "matchup": "", "team": "", "page": 1})
    pg = pd.DataFrame({"season": "2022-23", "game_id": "x", "game_date": d,
                       "player_id": pd.array([11, 22], dtype="Int64"), "player_name": ["P One", "P Two"],
                       "player_key": ["pone", "ptwo"], "team_id": pd.array([1, 4], dtype="Int64"),
                       "minutes": 10.0})
    pa, ts = av.build_availability(g, sel, parsed, pg)
    by = pa.set_index("player_key")
    assert by.loc["pone", "player_id"] == 11 and by.loc["pone", "match_method"] == "season_team"
    assert by.loc["ptwo", "player_id"] == 22 and by.loc["ptwo", "match_method"] == "season_unique_name"
    assert pd.isna(by.loc["pgone", "player_id"])
    t = ts.set_index(["game_id", "team"])
    assert t.loc[(g.game_id[0], "AAA"), "snapshot_usable"] and t.loc[(g.game_id[0], "AAA"), "n_rows"] == 2
    assert not t.loc[(g.game_id[1], "DDD"), "snapshot_usable"]           # NOT YET SUBMITTED
    assert t.loc[(g.game_id[1], "CCC"), "snapshot_usable"] and t.loc[(g.game_id[1], "CCC"), "n_rows"] == 0
