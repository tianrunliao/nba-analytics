"""Strength models: Elo, EWMA, dynamic strength — hand checks and causality."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_games

from nba_sos.models.game_model import _platt
from nba_sos.ratings import dynamic as DY
from nba_sos.ratings import elo as E


def test_elo_first_update_hand_computed(mini_season):
    e = E.run_elo(mini_season, k=20, hca=100, carry=0.5).set_index("game_id")
    g0 = mini_season.iloc[0]                      # AAA 110 - 100 BBB, both 1500
    p = 1 / (1 + 10 ** (-100 / 400))
    assert e.loc[g0.game_id, "p_home_elo"] == pytest.approx(p)
    mult = (10 + 3) ** 0.8 / (7.5 + 0.006 * 100)
    delta = 20 * mult * (1 - p)
    nxt = mini_season[mini_season.home_team == "DDD"].iloc[0]     # DDD vs AAA on 10-23
    assert e.loc[nxt.game_id, "away_elo"] == pytest.approx(1500 + delta)


def test_elo_season_carryover():
    g = make_games([("2021-22", "2022-01-01", "AAA", "BBB", 130, 90),
                    ("2022-23", "2022-11-01", "AAA", "BBB", 100, 99)])
    e = E.run_elo(g, k=20, hca=0, carry=0.5).set_index("game_id")
    r_end = 1500 + 20 * (40 + 3) ** 0.8 / 7.5 * 0.5
    assert e.iloc[1].home_elo == pytest.approx(1500 + 0.5 * (r_end - 1500))


@pytest.mark.parametrize("fn", [
    lambda g: E.run_elo(g, 20, 75, 0.65).rename(columns={"home_elo": "a", "away_elo": "b"})[["game_id", "a", "b"]],
    lambda g: DY.ewma_pregame(g, 14).rename(columns={"ewma_net": "a"}),
    lambda g: DY.dynamic_pregame(g, 30, 5, 0.7).rename(columns={"dyn_strength": "a"}),
])
def test_pregame_ratings_ignore_future_results(mini_season, fn):
    d = pd.Timestamp("2022-10-24").date()
    base = fn(mini_season)
    g2 = mini_season.copy()
    later = g2.game_date >= d
    g2.loc[later, ["home_score", "away_score"]] = [40, 160]
    g2.loc[later, "net_rating_margin"] = -120.0
    pert = fn(g2)
    early = mini_season[mini_season.game_date <= d].game_id
    a = base[base.game_id.isin(early)].sort_values(list(base.columns[:2])).reset_index(drop=True)
    b = pert[pert.game_id.isin(early)].sort_values(list(pert.columns[:2])).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_ewma_hand_computed():
    g = make_games([("2022-23", "2022-11-01", "AAA", "BBB", 110, 100),
                    ("2022-23", "2022-11-08", "AAA", "CCC", 100, 120),
                    ("2022-23", "2022-11-15", "AAA", "DDD", 100, 100 + 1)])
    e = DY.ewma_pregame(g, 7.0)
    aaa = e[e.team_id == 1].set_index("game_id").ewma_net
    assert np.isnan(aaa.iloc[0])
    assert aaa.iloc[1] == pytest.approx(10.0)
    w1 = np.exp(-1.0)                                   # game 1 is 7 days older than game 2
    assert aaa.iloc[2] == pytest.approx((10 * w1 + (-20)) / (w1 + 1))


def test_dynamic_strength_starts_at_prior_and_learns():
    prev = make_games([("2021-22", f"2022-01-{d:02d}", "AAA", "BBB", 120, 100) if d % 2 else
                       ("2021-22", f"2022-01-{d:02d}", "BBB", "AAA", 100, 120) for d in range(1, 11)])
    cur = make_games([("2022-23", "2022-11-01", "AAA", "BBB", 100, 100 + 1),
                      ("2022-23", "2022-11-03", "BBB", "AAA", 100, 90)])
    cur["game_id"] = ["c1", "c2"]
    d = DY.dynamic_pregame(pd.concat([prev, cur], ignore_index=True), 30, 5, 0.5)
    first = d[d.game_id == "c1"].set_index("team_id").dyn_strength
    assert first[1] > 0 > first[2]                        # prior from AAA's dominant last season
    second = d[d.game_id == "c2"].set_index("team_id").dyn_strength
    assert second[1] < first[1]                           # AAA lost game c1 -> rating falls


def test_platt_recovers_logistic_parameters():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 10, 20000)
    won = (rng.random(20000) < 1 / (1 + np.exp(-(0.15 * m + 0.2)))).astype(float)
    a, b = _platt(m, won)
    assert a == pytest.approx(0.15, abs=0.01) and b == pytest.approx(0.2, abs=0.05)
