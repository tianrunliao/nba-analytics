"""Ratings: rosters/player values, adjusted team strength, playoff model."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_games

from nba_sos.config import PROJECT_ROOT, Paths
from nba_sos.ratings import players as P
from nba_sos.ratings import playoff_model as PM
from nba_sos.ratings import team_strength as T


def _pg(rows):
    d = pd.DataFrame(rows, columns=["game_id", "player_id", "team_id", "minutes", "plus_minus"])
    d["player_id"], d["team_id"] = d.player_id.astype("Int64"), d.team_id.astype("Int64")
    d["season"] = "2022-23"
    return d


def test_roster_stints_trade_and_season_ending_absence():
    g = make_games([("2022-23", "2022-11-01", "AAA", "BBB", 100, 90),
                    ("2022-23", "2022-11-03", "AAA", "CCC", 100, 90),
                    ("2022-23", "2022-11-05", "AAA", "DDD", 100, 90),
                    ("2022-23", "2022-11-07", "BBB", "AAA", 100, 90)])
    gid = g.game_id.tolist()
    pg = _pg([(gid[0], 10, 1, 30, 5), (gid[1], 10, 1, 30, 5),          # p10: AAA, then injured for good
              (gid[0], 20, 1, 20, 3), (gid[3], 20, 2, 25, 4)])          # p20: traded AAA -> BBB
    pg = pg.merge(g[["game_id", "game_date"]], on="game_id")
    rg = P.roster_games(pg, g)
    p10 = rg[rg.player_id == 10].sort_values("game_date")
    assert p10.played.tolist() == [True, True, False, False]          # missed games counted to season end
    assert p10.stint_to_season_end.all()
    p20 = rg[rg.player_id == 20]
    aaa = p20[p20.team_id == 1].sort_values("game_date")
    assert aaa.game_date.max() < pd.Timestamp("2022-11-07")           # AAA stint ends before first BBB game
    assert not aaa.stint_to_season_end.any()
    assert p20[p20.team_id == 2].stint_to_season_end.all()


def test_on_off_value_hand_computed():
    g = make_games([("2022-23", "2022-11-01", "AAA", "BBB", 110, 100),
                    ("2022-23", "2022-11-03", "AAA", "CCC", 90, 100)])
    gid = g.game_id.tolist()
    pg = _pg([(gid[0], 10, 1, 24, 12)])     # played half of game 1 at +12, missed game 2
    pg = pg.merge(g[["game_id", "game_date"]], on="game_id")
    rg = P.roster_games(pg, g)
    v = P.player_values(rg, shrink_minutes=0).set_index("player_id").loc[10]
    # on: +12 in 24 min -> +24/48 ; off: (10-12) in 24 + (-10) in 48 -> -12 over 72 min -> -8/48
    assert v.on_off48 == pytest.approx(24 - (-8))
    assert v.mpg == 24 and v.games_played == 1
    hi = P.health_index(rg, P.player_values(rg, shrink_minutes=0))
    row = hi[(hi.game_id == gid[1]) & (hi.team_id == 1)].iloc[0]
    assert row.hi_minutes == pytest.approx(0.5) and row.hi_value == pytest.approx(0.5 * 32)


def test_pooled_ridge_recovers_strength_hca_and_rest_effect():
    rng = np.random.default_rng(1)
    true = {1: 6.0, 2: 2.0, 3: -2.0, 4: -6.0}
    rows = []
    for k in range(3000):
        h, a = rng.choice(list(true), 2, replace=False)
        b2b_h, b2b_a = rng.random() < 0.2, rng.random() < 0.2
        dm, dv = rng.normal(0, 0.6), rng.normal(0, 2.0)
        y = true[h] - true[a] + 3.0 - 2.0 * (b2b_h - b2b_a) - 1.0 * dm - 0.7 * dv + rng.normal(0, 12)
        rows.append({"season": "2022-23", "home_team_id": h, "away_team_id": a, "y": y, "home_flag": 1.0,
                     "d_b2b": float(b2b_h) - float(b2b_a), "d_hi_minutes": dm, "d_hi_value": dv,
                     "game_date": pd.Timestamp("2022-11-01") + pd.to_timedelta(int(k // 15), unit="D")})
    d = pd.DataFrame(rows)
    d["home_key"] = d.season + "|" + d.home_team_id.astype(str)
    d["away_key"] = d.season + "|" + d.away_team_id.astype(str)
    fit = T.fit_pooled(d, alpha=1.0)
    th = fit["theta"].set_index("team_id").theta
    for t, v in true.items():
        assert th[t] == pytest.approx(v, abs=0.8)
    assert fit["coefs"]["hca_2022-23"] == pytest.approx(3.0, abs=0.6)
    assert fit["coefs"]["b_b2b"] == pytest.approx(-2.0, abs=0.8)
    assert fit["coefs"]["b_val"] == pytest.approx(-0.7, abs=0.25)
    # no late-season change in the data -> momentum close to zero
    rec = T.fit_recent(d, fit, tau_days=20.0)
    assert rec.momentum.abs().max() < 2.0


def test_series_probability_properties():
    assert PM.series_win_prob(0, 0, True, 0.0, 14) == pytest.approx(0.5)
    assert PM.series_win_prob(0, 0, True, 4.0, 14) > 0.5 > PM.series_win_prob(0, 0, False, 4.0, 14)
    p = PM.series_win_prob(5, 0, True, 4.0, 14)
    assert 0.5 < p < 1 and PM.series_win_prob(5, 0, True, 4.0, 14) == pytest.approx(1 - PM.series_win_prob(0, 5, False, 4.0, 14))


def test_health_start_uses_status_and_rotation():
    roster = pd.DataFrame({"player_id": [1, 2, 3], "team_id": [9, 9, 9], "mpg": [36.0, 30.0, 8.0],
                           "value": [5.0, 0.0, -1.0]})
    statuses = pd.DataFrame({"player_id": [1, 2, 3], "team_id": [9, 9, 9],
                             "injury_status": ["Out", "Available", "Available"]})
    mt = pd.DataFrame({"injury_status": ["Out", "Out", "Available", "Available"], "rotation": [True, False, True, False],
                       "p_miss": [1.0, 1.0, 0.02, 0.36]})
    coefs = {"b_min": -1.0, "b_val": -0.5}
    team, det = PM.health_start(roster, statuses, None, mt, coefs)
    d = det.set_index("player_id").p_miss
    assert d[1] == 1.0 and d[2] == pytest.approx(0.02) and d[3] == pytest.approx(0.36)
    exp_min = 36 / 48 + 0.02 * 30 / 48 + 0.36 * 8 / 48
    exp_val = 36 / 48 * 5 + 0 + 0.36 * 8 / 48 * -1
    assert team.h_start.iloc[0] == pytest.approx(-1.0 * exp_min - 0.5 * exp_val)
    # proxy (no report): rotation players who missed the last k regular-season games in a row
    sk = pd.DataFrame({"player_id": [1, 2, 3], "team_id": [9, 9, 9], "streak": [6, 2, 9]})
    team2, det2 = PM.health_start(roster, None, sk, None, coefs, streak_k=5)
    assert det2.set_index("player_id").p_miss.tolist() == [1.0, 0.0, 0.0]   # short rest / bench ignored
    assert (det2.source == "rs_end_absence").all()


def test_rs_end_absence_streak_counts_trailing_missed_games_only():
    d = pd.date_range("2020-04-01", periods=6)
    rg = pd.DataFrame({"season": "2019-20", "player_id": [1] * 6 + [2] * 6 + [3] * 3, "team_id": 9,
                       "game_date": list(d) * 2 + list(d[3:]),
                       "played": [True, False, True, False, False, False] + [True] * 6 + [False] * 3,
                       "stint_to_season_end": [True] * 12 + [False] * 3})
    s = PM.rs_end_absence_streak(rg, "2019-20").set_index("player_id").streak
    assert s[1] == 3 and s[2] == 0 and 3 not in s.index        # traded-away stint excluded
    rg.loc[rg.player_id == 2, "played"] = False
    assert PM.rs_end_absence_streak(rg, "2019-20").set_index("player_id").streak[2] == 6


def test_choose_streak_threshold_maximises_agreement_with_report():
    x = pd.DataFrame({"streak": [0, 1, 1, 2, 4, 6, 8, 0], "out": [False, False, False, False, True, True, True, True]})
    rule = PM.choose_streak_threshold(x)
    assert rule["k"] in (3, 4) and rule["f1"][rule["k"]] == pytest.approx(2 * 1 * 0.75 / 1.75)
    assert rule["f1"][1] < rule["f1"][rule["k"]]


def test_playoff_improve_predictions_ignore_inplay_absences():
    from nba_sos.ratings import playoff_improve as PI
    rows = pd.DataFrame({"d_theta_full": [2.0, -2.0], "d_h_inplay": [-50.0, 50.0], "home_flag": [1.0, 1.0],
                         "home_win": [1, 0]})
    cols = ["d_theta_full", "d_h_inplay", "home_flag"]
    b = np.array([1.0, 1.0, 0.0])
    assert PI._logloss(rows, cols, b, 10.0) == pytest.approx(PI._logloss(rows.assign(d_h_inplay=0.0), cols, b, 10.0))


def test_fit_playoff_recovers_coefficients():
    rng = np.random.default_rng(2)
    n = 3000
    r = pd.DataFrame({c: rng.normal(0, 4, n) for c in ["d_theta_full", "d_momentum", "d_h_start", "d_h_inplay"]})
    r["home_flag"] = 1.0
    r["y"] = 1.4 * r.d_theta_full + 0.0 * r.d_momentum + 0.8 * r.d_h_start + 1.0 * r.d_h_inplay + 4 + rng.normal(0, 14, n)
    pm = PM.fit_playoff(r)
    assert pm["coef"]["d_theta_full"] == pytest.approx(1.4, abs=0.15)
    assert pm["coef"]["home_flag"] == pytest.approx(4.0, abs=0.7)
    teams = pd.DataFrame({"theta_full": [1.0], "momentum": [2.0], "h_start": [-1.0]})
    c = pm["coef"]
    assert PM.rating(teams, pm).iloc[0] == pytest.approx(c["d_theta_full"] - c["d_h_start"] + 2 * c["d_momentum"])


PATHS = Paths(PROJECT_ROOT)


@pytest.mark.skipif(not (PATHS.processed / "playoff_games.parquet").exists(), reason="run build first")
def test_target_season_playoff_results_do_not_change_its_ranking():
    """Scramble 2024-25 playoff results: 2024-25 ratings must not move (only the post-hoc
    performance column may)."""
    from nba_sos.config import load_configs
    from nba_sos.ratings import playoff_report as R
    cfg = load_configs(PATHS)
    hist = cfg["data_sources"]["seasons"]["historical"]
    target = "2024-25"
    hist_cut = hist[: hist.index(target) + 1]
    cfg = {**cfg, "data_sources": {**cfg["data_sources"], "seasons": {**cfg["data_sources"]["seasons"], "historical": hist_cut}}}
    d = R._load(PATHS)
    _, rank_a, _ = R.evaluate_and_rank(R.season_blocks(PATHS, cfg, d), hist_cut)
    pg = d["playoff_games"].copy()
    m = pg.season == target
    pg.loc[m, ["home_score", "away_score"]] = pg.loc[m, ["away_score", "home_score"]].to_numpy()
    pg.loc[m, "net_rating_margin"] = -pg.loc[m, "net_rating_margin"]
    pg.loc[m, "home_win"] = 1 - pg.loc[m, "home_win"]
    d["playoff_games"] = pg
    _, rank_b, _ = R.evaluate_and_rank(R.season_blocks(PATHS, cfg, d), hist_cut)
    a = rank_a[target].set_index("team_id").rating
    b = rank_b[target].set_index("team_id").rating.reindex(a.index)
    pd.testing.assert_series_equal(a, b)


def test_scale_blocks_divides_per100_quantities_by_season_noise():
    from nba_sos.ratings.playoff_report import scale_blocks
    rows = pd.DataFrame({"d_theta_full": [4.0], "d_h_inplay": [-2.0], "y": [12.0], "home_flag": [1.0]})
    teams = pd.DataFrame({"team_id": [1], "theta_full": [6.0], "momentum": [1.0], "h_start": [-1.0]})
    b = {"S": {"rows": {20.0: rows}, "teams": {20.0: teams}, "team_cols": ["theta_full", "momentum", "h_start"],
               "season_sigma": 12.0, "raw_net": pd.Series(dtype=float)}}
    s = scale_blocks(b)["S"]
    r = s["rows"][20.0].iloc[0]
    assert r.d_theta_full == pytest.approx(4 / 12) and r.y == pytest.approx(1.0) and r.home_flag == 1.0
    assert s["teams"][20.0].theta_full.iloc[0] == pytest.approx(0.5)
    assert rows.y.iloc[0] == 12.0                              # input untouched


def test_rating_uses_all_team_level_coefficients():
    pm = {"coef": {"d_theta_full": 1.4, "d_star": 0.5, "d_h_inplay": 9.9, "home_flag": 4.0}}
    teams = pd.DataFrame({"theta_full": [2.0], "star": [1.0]})
    assert PM.rating(teams, pm).iloc[0] == pytest.approx(1.4 * 2 + 0.5)
