"""Kalman-filter strength: hand-computed update, recovery on synthetic data, season carry,
and agreement between the pre-game output and the as-of query."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_games

from nba_sos.ratings import kalman as KF


def _league(true_s: dict[int, float], n_dates: int, seed: int) -> pd.DataFrame:
    """One synthetic season: every date, the teams are paired at random; margin = S_h - S_a + noise."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(true_s))
    rows = []
    for d in pd.date_range("2022-10-20", periods=n_dates, freq="D"):
        perm = rng.permutation(ids)
        for h, a in zip(perm[0::2], perm[1::2]):
            rows.append(
                {
                    "season": "2022-23",
                    "game_id": f"g{len(rows):05d}",
                    "game_date": d.date(),
                    "home_team_id": int(h),
                    "away_team_id": int(a),
                    "is_neutral": False,
                    "net_rating_margin": true_s[h] - true_s[a] + rng.normal(0.0, 10.0),
                }
            )
    return pd.DataFrame(rows)


def test_first_update_hand_computed():
    g = make_games(
        [("2022-23", "2022-10-20", "AAA", "BBB", 110, 100), ("2022-23", "2022-10-22", "AAA", "BBB", 100, 100)]
    )
    p0, r = 20.0, 12.0
    k = KF.kalman_pregame(g, q=0.0, p0=p0, carry=0.7, obs_sd=r)
    first = k[k.game_id == g.game_id[0]].set_index("team_id")
    assert first.kal_strength.tolist() == [0.0, 0.0]
    assert first.kal_sd.tolist() == pytest.approx([np.sqrt(p0 / 2)] * 2)  # two teams: P = p0 * (I - 11'/2)
    y = float(g.net_rating_margin[0])  # first season: home court 0
    second = k[k.game_id == g.game_id[1]].set_index("team_id")
    gain = p0 / (2 * p0 + r**2)
    assert second.kal_strength[1] == pytest.approx(gain * y)
    assert second.kal_strength[2] == pytest.approx(-gain * y)
    assert second.kal_sd[1] == pytest.approx(np.sqrt(p0 / 2 - p0 * gain))


def test_recovers_synthetic_strengths():
    true_s = {1: -6.0, 2: -3.0, 3: -1.0, 4: 1.0, 5: 3.0, 6: 6.0}
    g = _league(true_s, n_dates=150, seed=0)
    fs = KF.filter_states(g, {"q": 0.001, "p0": 50.0, "carry": 0.7, "obs_sd": 10.0})
    st = KF.strengths_asof(fs, "2022-23", [pd.Timestamp("2023-06-01")]).set_index("team_id")
    est = st.S - st.S.mean()
    truth = pd.Series(true_s)
    assert np.abs(est - truth).max() < 1.5
    assert np.corrcoef(est, truth)[0, 1] > 0.97
    assert st.S.sum() == pytest.approx(0.0, abs=1e-9)  # relative to the league average
    assert (st.sd < np.sqrt(50.0) / 5).all()  # uncertainty shrinks with data


def test_season_carry_and_new_season_variance():
    prev = _league({1: -5.0, 2: 0.0, 3: 2.0, 4: 3.0}, n_dates=40, seed=1).assign(season="2021-22")
    cur = _league({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, n_dates=5, seed=2)
    cur["game_date"] = [d.replace(year=d.year + 1) for d in cur.game_date]
    cur["game_id"] = "c" + cur.game_id
    par = {"q": 0.01, "p0": 10.0, "carry": 0.6, "obs_sd": 12.0}
    fs = KF.filter_states(pd.concat([prev, cur], ignore_index=True), par)
    end_prev = fs["seasons"]["2021-22"]
    start = KF.strengths_asof(fs, "2022-23", [pd.Timestamp("2023-10-01")])
    assert start.S.to_numpy() == pytest.approx(0.6 * end_prev["post_S"][-1])
    assert start.sd.to_numpy() ** 2 == pytest.approx(0.36 * end_prev["post_var"][-1] + 10.0 * (1 - 1 / 4))


def test_asof_matches_pregame_output(mini_season):
    par = {"q": 0.05, "p0": 10.0, "carry": 0.7, "obs_sd": 12.0}
    pre = KF.kalman_pregame(mini_season, **par)
    fs = KF.filter_states(mini_season, par)
    d = pd.Timestamp("2022-10-24")
    st = KF.strengths_asof(fs, "2022-23", [d]).set_index("team_id")
    today = mini_season[pd.to_datetime(mini_season.game_date) == d].game_id
    p = pre[pre.game_id.isin(today)].set_index("team_id")
    assert p.kal_strength.to_numpy() == pytest.approx(st.S.reindex(p.index).to_numpy())
    assert p.kal_sd.to_numpy() == pytest.approx(st.sd.reindex(p.index).to_numpy())


def test_asof_ignores_games_on_or_after_cutoff():
    rows = [
        ("2021-22", f"2021-11-{d:02d}", h, a, 100 + 3 * d % 7, 100)
        for d, (h, a) in zip(range(1, 13), [("AAA", "BBB"), ("CCC", "DDD"), ("AAA", "CCC"), ("BBB", "DDD")] * 3)
    ]
    rows += [
        ("2022-23", f"2022-11-{d:02d}", h, a, 100 + (5 * d) % 9, 100)
        for d, (h, a) in zip(range(1, 13), [("BBB", "AAA"), ("DDD", "CCC"), ("CCC", "AAA"), ("DDD", "BBB")] * 3)
    ]
    g = make_games(rows)
    par = {"q": 0.05, "p0": 10.0, "carry": 0.7, "obs_sd": 12.0}
    cut = pd.Timestamp("2022-11-07")
    a = KF.strengths_asof(KF.filter_states(g, par), "2022-23", [cut])
    g2 = g.copy()
    g2.loc[(g2.season == "2022-23") & (pd.to_datetime(g2.game_date) >= cut), "net_rating_margin"] = 50.0
    pd.testing.assert_frame_equal(a, KF.strengths_asof(KF.filter_states(g2, par), "2022-23", [cut]))
