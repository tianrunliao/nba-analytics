"""Player value modules: era standardisation, box calibration, lineup RAPM, star/depth."""
import numpy as np
import pandas as pd
import pytest

from nba_sos.ratings import lineup_rapm as L
from nba_sos.ratings import player_value as V
from nba_sos.ratings.player_report import min_minutes, team_star_depth


def _stats(season, n=40, seed=0, pts_shift=0.0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"season": season, "player_id": np.arange(n) + (0 if season == "A" else 1000),
                      "player_name": [f"p{i}" for i in range(n)], "min_total": rng.uniform(300, 2800, n),
                      "gp": 70, "ts": rng.normal(0.57, 0.04, n), "usg": rng.uniform(0.12, 0.33, n),
                      "pie": rng.normal(0.1, 0.03, n), "on_court_net": rng.normal(0, 4, n)})
    for c in V.PER100:
        d[f"{c.lower()}_100"] = rng.normal(10, 3, n)
    d["pts_100"] = rng.normal(22 + pts_shift, 6, n)
    return d


def test_standardise_removes_era_level_shift():
    """Same distribution of players, league scoring 9 points higher in season B -> same z."""
    a, b = _stats("A", seed=1), _stats("B", seed=1, pts_shift=9.0)
    onoff = pd.DataFrame({"season": [], "player_id": [], "on_off48": []})
    tn = pd.concat([a, b])[["season", "player_id"]].assign(team_net=0.0)
    z = V.standardise(pd.concat([a, b]), onoff, tn)
    za, zb = z[z.season == "A"].z_pts_100.to_numpy(), z[z.season == "B"].z_pts_100.to_numpy()
    assert np.allclose(za, zb)
    ref = z[(z.season == "A") & (z.min_total >= V.MIN_FOR_NORMS)]
    assert np.average(ref.z_pts_100, weights=ref.min_total) == pytest.approx(0.0, abs=1e-9)


def test_team_design_minute_shares_sum_to_five():
    pg = pd.DataFrame({"season": "A", "team_id": [1] * 6, "player_id": [1, 2, 3, 4, 5, 6],
                       "minutes": [400, 300, 300, 200, 200, 100.0], "game_id": "g"})
    z = pd.DataFrame({"season": "A", "player_id": [1, 2, 3, 4, 5, 6]})
    for f in V.FEATURES:
        z[f"z_{f}"] = 1.0
    d = V.team_design(z, pg)
    assert d[f"z_{V.FEATURES[0]}"].iloc[0] == pytest.approx(5.0)


def test_fit_weights_recovers_signal():
    rng = np.random.default_rng(3)
    n = 400
    des = pd.DataFrame({"season": "A", "team_id": np.arange(n)})
    for f in V.FEATURES:
        des[f"z_{f}"] = rng.normal(0, 2, n)
    theta = des[["season", "team_id"]].assign(theta=0.8 * des.z_pie - 0.5 * des.z_tov_100 + rng.normal(0, 0.5, n))
    w = V.fit_weights(des, theta)
    assert w["beta"]["pie"] == pytest.approx(0.8, abs=0.08) and w["beta"]["tov_100"] == pytest.approx(-0.5, abs=0.08)
    assert w["cv_r2"] > 0.8


def test_lineup_rapm_recovers_player_effects_and_uses_prior_when_helpful():
    rng = np.random.default_rng(4)
    n_players = 40
    true = rng.normal(0, 3, n_players)
    rows = []
    for _ in range(3000):
        five = rng.choice(n_players, 5, replace=False)
        mins = rng.uniform(5, 60)
        y = true[five].sum() + rng.normal(0, 40 / np.sqrt(mins))
        pm = y * mins / 48 * 100 / 100
        rows.append({"GROUP_ID": "-" + "-".join(str(p) for p in five) + "-", "MIN": mins, "PLUS_MINUS": pm})
    lu = pd.DataFrame(rows)
    good_prior = pd.Series(true + rng.normal(0, 0.5, n_players), index=range(n_players))
    out, info = L.fit_season(lu, pace=100.0, prior_by_player=good_prior)
    r = out.set_index("player_id").lpm.reindex(range(n_players)).to_numpy()
    assert np.corrcoef(r, true)[0, 1] > 0.9
    assert info["kappa"] > 0                      # an informative prior is used
    noise_prior = pd.Series(rng.normal(0, 3, n_players), index=range(n_players))
    _, info2 = L.fit_season(lu, pace=100.0, prior_by_player=noise_prior)
    assert info2["kappa"] <= info["kappa"]       # an uninformative prior gets less weight


def test_star_depth_definition():
    r = pd.DataFrame({"team_id": 1, "player_id": range(10),
                      "mpg": [36, 34, 32, 30, 28, 22, 20, 18, 16, 5],
                      "lpm": [6.0, 1.0, 4.0, 0.0, 0.0, -1.0, 1.0, 2.0, 0.0, 9.0]})
    sd = team_star_depth(r).iloc[0]
    assert sd.star == pytest.approx(5.0)          # two best among the top 8 by minutes: 6 and 4
    assert sd.depth == pytest.approx(np.average([-1.0, 1.0, 2.0, 0.0], weights=[22, 20, 18, 16]))


def test_min_minutes_prorated_for_short_seasons():
    g = pd.DataFrame({"season": "X", "home_team_id": [1, 2] * 36, "away_team_id": [2, 1] * 36})
    assert min_minutes("X", g) == pytest.approx(1500 * 72 / 82)


def test_lineup_box_weights_recover_feature_effect():
    """Lineup net = 2 * (sum of players' z_pts) + noise -> weight on pts ~ 2, others ~ 0."""
    rng = np.random.default_rng(7)
    n_p = 60
    feats = ["pts_100", "ast_100"]
    z = pd.DataFrame({"season": "S", "player_id": np.arange(n_p),
                      "z_pts_100": rng.normal(0, 1, n_p), "z_ast_100": rng.normal(0, 1, n_p)})
    rows = []
    for _ in range(4000):
        five = rng.choice(n_p, 5, replace=False)
        mins = rng.uniform(5, 40)
        y = 2 * z.z_pts_100.iloc[five].sum() + rng.normal(0, 25 / np.sqrt(mins))
        rows.append({"GROUP_ID": "-" + "-".join(map(str, five)) + "-", "MIN": mins, "PLUS_MINUS": y * mins / 48,
                     "TEAM_ID": int(rng.integers(0, 10)), "season": "S"})
    lu = pd.DataFrame(rows)
    for demean in (False, True):
        fb = L.fit_box_on_lineups([L.lineup_box_design(lu, 100.0, z, feats, demean=demean)], feats, alpha=10.0)
        assert fb["beta"]["pts_100"] == pytest.approx(2.0, abs=0.2)
        assert abs(fb["beta"]["ast_100"]) < 0.2
