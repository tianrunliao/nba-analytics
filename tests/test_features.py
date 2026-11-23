"""Pre-game features on the synthetic mini season (see conftest for the 8 games)."""
import numpy as np
import pandas as pd
import pytest
from conftest import EMPTY_AVAIL, EMPTY_PLAYER_GAMES, FEATURE_CFG, team_status_all_unusable

from nba_sos.features.pregame import build_team_game_features, haversine_km


def features(games, venues, avail=None, status=None, pg=None):
    return build_team_game_features(games, venues, EMPTY_AVAIL if avail is None else avail,
                                    team_status_all_unusable(games) if status is None else status,
                                    EMPTY_PLAYER_GAMES if pg is None else pg, FEATURE_CFG)


def row(f, team, date):
    r = f[(f.team == team) & (f.game_date == pd.Timestamp(date).date())]
    assert len(r) == 1
    return r.iloc[0]


def test_efficiency_to_date_uses_only_earlier_games(mini_season, venues):
    f = features(mini_season, venues)
    r = row(f, "AAA", "2022-10-24")       # prior: W 110-100 (H), L 99-101 (A)
    assert r.season_games_played == 2
    assert r.season_win_pct_to_date == 0.5
    assert r.season_off_rating_to_date == pytest.approx(104.5)
    assert r.season_def_rating_to_date == pytest.approx(100.5)
    assert r.season_net_rating_to_date == pytest.approx(4.0)
    assert r.season_avg_margin_to_date == pytest.approx(4.0)
    assert r.season_pace_to_date == pytest.approx(100.0)
    first = row(f, "AAA", "2022-10-20")
    assert first.season_games_played == 0 and np.isnan(first.season_net_rating_to_date)


def test_schedule_density(mini_season, venues):
    f = features(mini_season, venues)
    assert row(f, "BBB", "2022-10-21").is_back_to_back
    r = row(f, "AAA", "2022-10-24")
    assert r.rest_days == 0 and r.is_back_to_back
    assert r.games_last_4_days == 2 and r.games_last_7_days == 3   # includes current game
    assert row(f, "AAA", "2022-10-23").rest_days == 2
    assert pd.isna(row(f, "AAA", "2022-10-20").rest_days)          # season opener: not imputed


def test_travel_distance_timezone_and_road_runs(mini_season, venues):
    f = features(mini_season, venues)
    opener = row(f, "AAA", "2022-10-20")
    assert opener.travel_distance_km == 0 and opener.travel_origin == "season_start_home_venue"
    at_la = row(f, "AAA", "2022-10-23")   # from (0,0)/UTC to Los Angeles
    assert at_la.travel_distance_km == pytest.approx(haversine_km(0.0, 0.0, 34.0430, -118.2672))
    assert at_la.timezone_change_hours == -7      # PDT in October, moving west
    assert row(f, "AAA", "2022-10-24").timezone_change_hours == 7
    assert row(f, "AAA", "2022-10-23").consecutive_road_games == 1
    assert row(f, "AAA", "2022-10-24").consecutive_road_games == 0
    # neutral site counts as road for the nominal home team too: A, H, A, neutral
    assert list(f[f.team == "DDD"].sort_values("game_date").consecutive_road_games) == [1, 0, 1, 2]
    # BBB: 1 km of longitude at the equator between V_AAA and V_BBB
    assert row(f, "BBB", "2022-10-21").travel_distance_km == pytest.approx(111.2, abs=0.2)


def test_haversine_known_distance():
    # New York (MSG) to Los Angeles (Crypto.com) is about 3,940 km great-circle
    assert haversine_km(40.7505, -73.9935, 34.0430, -118.2672) == pytest.approx(3940, abs=15)


def test_no_leakage_from_future_results(mini_season, venues):
    """Changing every result on or after D must not change any feature of a game on or before D."""
    d = pd.Timestamp("2022-10-24").date()
    base = features(mini_season, venues)
    g2 = mini_season.copy()
    future = g2.game_date >= d
    g2.loc[future, "home_score"] = 50
    g2.loc[future, "away_score"] = 150
    g2.loc[future, "possessions"] = 80.0
    pert = features(g2, venues)
    cols = [c for c in base.columns if c.startswith("season_")]
    keep = base.game_date <= d
    pd.testing.assert_frame_equal(base.loc[keep, cols], pert.loc[keep, cols])
    # and the perturbation is visible afterwards (sanity check that the test can fail)
    assert not base.loc[~keep, cols].equals(pert.loc[~keep, cols])


def test_as_of_timestamp_precedes_tipoff(mini_season, venues):
    f = features(mini_season, venues)
    r = row(f, "AAA", "2022-10-24")
    assert r.results_cutoff_utc == pd.Timestamp("2022-10-24 03:00", tz="UTC")  # 10-23 23:00 + 4h
    assert (f.as_of_timestamp.dropna() < f.tipoff_utc[f.as_of_timestamp.notna()]).all()


def _injury_inputs(games):
    g4 = games[(games.game_date == pd.Timestamp("2022-10-24").date()) & (games.home_team == "AAA")].game_id.iloc[0]
    by_date = dict(zip(games.game_date.astype(str) + games.home_team, games.game_id))
    avail = pd.DataFrame({
        "season": "2022-23", "game_id": g4, "team_id": pd.array([1, 1, 1], dtype="Int64"), "team": "AAA",
        "player_id": pd.array([101, 102, pd.NA], dtype="Int64"),
        "player_name_raw": ["P, One", "P, Two", "Never, Played"], "player_key": ["onep", "twop", "playednever"],
        "injury_status": ["Out", "Questionable", "Out"], "status_valid": True, "reason": "",
        "report_path": "r.pdf", "report_timestamp": pd.Timestamp("2022-10-24 22:30", tz="UTC"),
        "match_method": ["season_team", "season_team", None], "source": "test"})
    status = team_status_all_unusable(games)
    m = (status.game_id == g4) & (status.team_id == 1)
    status.loc[m, ["snapshot_usable", "report_found", "game_in_report"]] = True
    status.loc[m, "report_published_at_utc"] = pd.Timestamp("2022-10-24 22:30", tz="UTC")
    pg = pd.DataFrame({
        "season": "2022-23",
        "game_id": [by_date["2022-10-20AAA"], by_date["2022-10-23DDD"], by_date["2022-10-26CCC"]],
        "game_date": [pd.Timestamp(x).date() for x in ["2022-10-20", "2022-10-23", "2022-10-26"]],
        "player_id": pd.array([101, 101, 101], dtype="Int64"), "player_name": "One P", "player_key": "onep",
        "team_id": pd.array([1, 1, 1], dtype="Int64"), "minutes": [30.0, 20.0, 40.0]})
    return avail, status, pg


def test_injury_features_use_prior_minutes_and_stay_nan_without_snapshot(mini_season, venues):
    avail, status, pg = _injury_inputs(mini_season)
    f = features(mini_season, venues, avail, status, pg)
    r = row(f, "AAA", "2022-10-24")
    assert r.injury_snapshot_usable
    assert r.n_out == 2 and r.n_questionable == 1 and r.n_missing_players == 2
    assert r.missing_minutes_prior_avg == pytest.approx(25.0)   # (30+20)/2, the 10-26 game excluded
    assert r.n_missing_unmatched == 1
    assert r.as_of_timestamp == pd.Timestamp("2022-10-24 22:30", tz="UTC")
    other = row(f, "CCC", "2022-10-24")
    assert not other.injury_snapshot_usable and np.isnan(other.n_out)   # not imputed as 0
    # future minutes do not leak into the earlier snapshot
    pg2 = pg.copy()
    pg2.loc[2, "minutes"] = 0.0
    f2 = features(mini_season, venues, avail, status, pg2)
    assert row(f2, "AAA", "2022-10-24").missing_minutes_prior_avg == pytest.approx(25.0)


def test_back_to_back_context_late_west_then_east_both_away(venues):
    """AAA: home 10-31, away at DDD (Los Angeles) 11-01 late, away at CCC (New York) 11-02."""
    from conftest import make_games
    g = make_games([
        ("2022-23", "2022-10-31", "AAA", "BBB", 100, 90),
        ("2022-23", "2022-11-01", "DDD", "AAA", 100, 90),
        ("2022-23", "2022-11-02", "CCC", "AAA", 100, 90),
        ("2022-23", "2022-11-02", "BBB", "DDD", 100, 90),
    ])
    g.loc[1, "tipoff_utc"] = pd.Timestamp("2022-11-02 02:30", tz="UTC")   # 22:30 ET, 19:30 PDT
    g.loc[1, "overtime_periods"] = 1
    f = features(g, venues)
    r = row(f, "AAA", "2022-11-02")
    assert r.b2b_type == "AA" and r.leg == "A"
    assert r.turnaround_hours == pytest.approx(20.5)
    assert r.travel_distance_km == pytest.approx(haversine_km(34.0430, -118.2672, 40.7505, -73.9935))
    assert r.timezone_change_hours == 3                     # flying east
    assert r.prev_tipoff_local_hour == pytest.approx(19.5)  # Los Angeles wall clock
    assert r.tipoff_local_hour == pytest.approx(19.0)       # New York wall clock
    assert r.body_clock_tipoff_hour == pytest.approx(23.0)  # AAA's home zone is UTC
    assert r.prev_overtime_periods == 1
    assert row(f, "AAA", "2022-11-01").b2b_type == "HA"
    assert pd.isna(row(f, "AAA", "2022-10-31").b2b_type)    # opener: no previous game
    assert pd.isna(row(f, "BBB", "2022-11-02").b2b_type)    # BBB rested a day
