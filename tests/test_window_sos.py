import numpy as np
import pandas as pd
import pytest

from nba_sos.features.pregame import team_game_long
from nba_sos.sos import window

PARAMS = {"g": 1.0, "hca": 2.0, "sigma": 14.0}


def test_reference_win_prob_directions():
    p = window.reference_win_prob(np.array([0.0, 0.0, 0.0, 5.0]), np.array([0, 1, -1, 0]), PARAMS)
    assert p[0] == pytest.approx(0.5)
    assert p[1] > 0.5 > p[2] and p[3] < 0.5
    assert window.reference_win_prob(np.array([0.0]), np.array([1]), PARAMS, home_court=False)[0] == pytest.approx(0.5)


def _window(rows):
    return pd.DataFrame(rows, columns=["team_id", "team", "game_id", "opponent_id", "home_away", "is_neutral"])


def test_window_sos_positive_for_strong_opponents_and_home_court_split():
    w = _window([(1, "AAA", "g1", 3, "A", False), (1, "AAA", "g2", 3, "A", False),
                 (2, "BBB", "g3", 4, "H", False), (2, "BBB", "g4", 4, "H", False)])
    st = pd.DataFrame({"team_id": [3, 4], "S": [6.0, -6.0]})
    out = window.window_sos(w, st, PARAMS).set_index("team")
    assert out.loc["AAA", "d_sos"] > 0 > out.loc["BBB", "d_sos"]
    assert out.loc["AAA", "d_sos_rank"] == 1
    # two road games add difficulty, two home games remove it
    assert out.loc["AAA", "home_court_part"] > 0 > out.loc["BBB", "home_court_part"]
    exp = 1 - 2 * window.reference_win_prob(np.array([6.0]), np.array([-1]), PARAMS)[0]
    assert out.loc["AAA", "d_sos"] == pytest.approx(exp)


def test_window_sos_pairwise_strengths_and_missing_strength():
    w = _window([(1, "AAA", "g1", 3, "H", False), (2, "BBB", "g2", 3, "H", False)])
    pair = pd.DataFrame({"team_id": [1, 2], "opponent_id": [3, 3], "S": [4.0, -4.0]})
    out = window.window_sos(w, pair, PARAMS).set_index("team")
    assert out.loc["AAA", "opp_strength_mean"] == 4.0 and out.loc["BBB", "opp_strength_mean"] == -4.0
    with pytest.raises(ValueError):
        window.window_sos(w, pd.DataFrame({"team_id": [9], "S": [0.0]}), PARAMS)


def test_first_n_stats_split(mini_season):
    s = window.first_n_stats(team_game_long(mini_season), 2).set_index("team_id")
    # AAA first two: 110-100, 99-101 over 200 poss -> +4.0 ; rest: 100-102, 111-88 -> +10.5
    assert s.loc[1, "net_first_n"] == pytest.approx(4.0)
    assert s.loc[1, "net_rest"] == pytest.approx(10.5)
