"""Playoff feature additions: in-playoff form, rotation shift, experience, selection."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_games

from nba_sos.ratings import playoff_features as F
from nba_sos.ratings import playoff_improve as PI


def _po_games():
    g = make_games(
        [
            ("2022-23", "2023-04-15", "AAA", "BBB", 120, 100),
            ("2022-23", "2023-04-17", "AAA", "BBB", 100, 110),
            ("2022-23", "2023-04-20", "BBB", "AAA", 90, 100),
        ]
    )
    g["game_id"] = ["0042200101", "0042200102", "0042200103"]
    return g


def test_po_form_uses_only_earlier_playoff_games():
    g = _po_games()
    theta = pd.Series({1: 2.0, 2: 0.0})
    f = F.po_form(g, theta, hca=3.0).set_index("game_id")
    assert f.loc["0042200101", "home_po_form"] == 0.0 and f.loc["0042200101", "away_po_form"] == 0.0
    r1 = 20 - (2 + 3)  # game 1 residual for AAA
    assert f.loc["0042200102", "home_po_form"] == pytest.approx(r1 / (1 + F.FORM_PRIOR_GAMES))
    assert f.loc["0042200102", "away_po_form"] == pytest.approx(-r1 / (1 + F.FORM_PRIOR_GAMES))
    # changing game 3 cannot change the form entering games 1-2
    g2 = g.copy()
    g2.loc[2, "net_rating_margin"] = 99.0
    f2 = F.po_form(g2, theta, 3.0).set_index("game_id")
    pd.testing.assert_frame_equal(f.loc[["0042200101", "0042200102"]], f2.loc[["0042200101", "0042200102"]])


def test_playoff_shift_rewards_top_heavy_value():
    roster = pd.DataFrame(
        {
            "team_id": 1,
            "player_id": range(10),
            "rs_minutes": [2500, 2400, 2200, 2000, 1800, 1500, 1200, 1000, 600, 300.0],
            "value": [6.0, 3.0, 1.0, 0.0, 0.0, -1.0, -1.0, -2.0, -2.0, -3.0],
        }
    )
    tot = roster.rs_minutes.sum()
    rs_share = roster.rs_minutes / tot * 5
    curve = rs_share.to_numpy().copy()
    curve[:5] += 0.1  # playoffs: +0.1 share for the top five ...
    curve[5:] -= 0.1  # ... taken from the bench
    sh = F.playoff_shift(roster, curve, top=10).iloc[0].playoff_shift
    assert sh == pytest.approx(0.1 * (6 + 3 + 1 + 0 + 0) - 0.1 * (-1 - 1 - 2 - 2 - 3))
    assert F.playoff_shift(roster, None, top=10).iloc[0].playoff_shift == pytest.approx(0.0)


def test_experience_age_weighted_by_minutes():
    roster = pd.DataFrame({"team_id": [1, 1], "player_id": [10, 11], "rs_minutes": [3000.0, 1000.0], "value": 0.0})
    e = F.experience_age(roster, pd.Series({10: np.e - 1}), pd.Series({10: 30.0, 11: 22.0})).iloc[0]
    assert e.experience == pytest.approx(0.75 * 1.0 + 0.25 * 0.0)
    assert e.age == pytest.approx(0.75 * 30 + 0.25 * 22)


def test_forward_selection_keeps_only_groups_that_help_out_of_fold():
    rng = np.random.default_rng(0)
    train = {}
    for s in range(6):
        n = 300
        r = pd.DataFrame({c: rng.normal(0, 4, n) for c in PI.BASE if c != "home_flag"})
        r["home_flag"] = 1.0
        for cols in PI.GROUPS.values():
            for c in cols:
                r[c] = rng.normal(0, 1, n)
        r["y"] = 1.3 * r.d_theta_full + 3 + 4.0 * r.d_po_form + rng.normal(0, 14, n)
        r["home_win"] = (r.y > 0).astype(int)
        train[str(s)] = r
    cols, chosen, _ = PI.forward_select(train)
    assert chosen[:1] == ["po_form"] and "style" not in chosen
