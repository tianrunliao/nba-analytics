import numpy as np
import pandas as pd
import pytest
from conftest import make_games

from nba_sos.evaluation import metrics as M
from nba_sos.features import matchups
from nba_sos.models import preseason
from nba_sos.preprocessing.games import build_schedule
from nba_sos.sos.static import static_sos


# ------------------------------------------------------------------ static SOS
def test_retro_owp_excludes_games_against_the_team(mini_season):
    sos, _ = static_sos(mini_season)
    a = sos[sos.team == "AAA"].iloc[0]
    # opponents excl. games vs AAA: BBB 2/3, DDD 2/3, CCC 0/2 (twice)
    assert a.retro_owp == pytest.approx((2 / 3 + 2 / 3 + 0 + 0) / 4)
    assert sos.retro_owp_rank.between(1, 4).all()


def test_causal_sos_uses_only_opponent_games_before_the_date(mini_season):
    _, per_game = static_sos(mini_season)
    gid = mini_season[(mini_season.home_team == "AAA") & (mini_season.away_team == "CCC")].game_id.iloc[0]
    r = per_game[(per_game.game_id == gid) & (per_game.team_id == 1)].iloc[0]
    assert r.opp_win_pct_to_date_ex == 0.0  # CCC lost both earlier games
    assert r.opp_net_rating_to_date == pytest.approx(-20.0)
    opener = per_game[per_game.game_id == mini_season.game_id.iloc[0]]
    assert opener.opp_win_pct_to_date_ex.isna().all()

    g2 = mini_season.copy()
    later = g2.game_date > pd.Timestamp("2022-10-24").date()
    g2.loc[later, ["home_score", "away_score"]] = [0, 200]
    _, per_game2 = static_sos(g2)
    early = per_game.merge(mini_season[["game_id", "game_date"]], on="game_id")
    keep = early.game_date <= pd.Timestamp("2022-10-24").date()
    pd.testing.assert_frame_equal(
        per_game[keep.values].reset_index(drop=True), per_game2[keep.values].reset_index(drop=True)
    )


# ------------------------------------------------------------------ matchups
STANDINGS = pd.DataFrame(
    {
        "season": "2022-23",
        "team_id": pd.array([1, 2, 3, 4], dtype="Int64"),
        "conference": ["East", "East", "West", "West"],
        "division": ["Atlantic", "Atlantic", "Pacific", "Northwest"],
    }
)


def test_matchup_counts_symmetric_with_relations(mini_season, venues):
    s = build_schedule(mini_season, STANDINGS, venues)
    c = matchups.matchup_counts(s)
    assert matchups.symmetry_violations(c) == 0
    m = matchups.matrix(c, "2022-23")
    assert m.loc["AAA", "CCC"] == 2 and m.loc["CCC", "AAA"] == 2 and m.loc["AAA", "AAA"] == 0
    rel = c.set_index(["team", "opponent"]).relation
    assert rel["AAA", "BBB"] == "same_division"
    assert rel["CCC", "DDD"] == "same_conference_other_division"
    assert rel["AAA", "DDD"] == "other_conference"
    assert int(m.sum(axis=1)["AAA"]) == 4


def test_structure_deviation_detection():
    def team_counts(profile):
        rows = []
        for rel, d in profile.items():
            for meetings, n_opp in d.items():
                rows += [{"season": "s", "team": "T", "relation": rel, "n_games": meetings}] * n_opp
        return pd.DataFrame(rows)

    assert len(matchups.structure_deviations(team_counts(matchups.TRADITIONAL))) == 0
    cup = {**matchups.TRADITIONAL, "same_division": {4: 3, 5: 1}}
    dev = matchups.structure_deviations(team_counts(cup))
    assert dev.relation.tolist() == ["same_division"]


# ------------------------------------------------------------------ metrics & splits
def test_metric_values():
    y, p = np.array([1, 0]), np.array([0.8, 0.3])
    assert M.log_loss(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert M.brier(y, p) == pytest.approx(0.065)
    assert M.mae([1, 3], [2, 1]) == 1.5 and M.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    cal = M.calibration_table(np.array([1, 0, 1, 1]), np.array([0.05, 0.05, 0.95, 0.95]))
    assert cal.n.tolist() == [2, 2] and cal.observed.tolist() == [0.5, 1.0]


def test_rolling_splits_are_strictly_earlier():
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    splits = M.rolling_splits(seasons, "2021-22", ["2022-23", "2024-25"])
    assert splits == [(["2021-22"], "2022-23"), (["2021-22", "2022-23", "2023-24"], "2024-25")]
    with pytest.raises(ValueError):
        M.rolling_splits(seasons, "2021-22", ["2021-22"])


# ------------------------------------------------------------------ baseline
def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    d = pd.DataFrame({"prior_diff": rng.normal(0, 6, n), "is_neutral": rng.random(n) < 0.05})
    d["home_flag"] = (~d.is_neutral).astype(float)
    d["margin"] = 0.5 * d.prior_diff + 3.0 * d.home_flag + rng.normal(0, 12, n)
    d["home_win"] = (d.margin > 0).astype(int)
    p = preseason.fit(d)
    assert p["b"] == pytest.approx(0.5, abs=0.05) and p["hca"] == pytest.approx(3.0, abs=0.5)
    assert p["sigma"] == pytest.approx(12, abs=0.5)


def test_prior_ratings_come_from_previous_season_only():
    prev = make_games(
        [("2021-22", "2021-11-01", "AAA", "BBB", 120, 100), ("2021-22", "2021-11-02", "CCC", "DDD", 100, 100 + 10)]
    )
    cur = make_games([("2022-23", "2022-11-01", "AAA", "CCC", 90, 100)])
    pr = preseason.prior_ratings(pd.concat([prev, cur]))
    a = pr[(pr.season == "2022-23") & (pr.team_id == 1)].prior_net_rating.iloc[0]
    assert a == pytest.approx(20.0)
    cur2 = cur.assign(home_score=pd.array([10], dtype="Int64"))
    pr2 = preseason.prior_ratings(pd.concat([prev, cur2]))
    assert pr2[pr2.season == "2022-23"].equals(pr[pr.season == "2022-23"])


def test_neutral_games_get_no_home_advantage():
    params = {"b": 0.5, "hca": 2.0, "sigma": 13.0, "const_home_win_rate": 0.56, "const_home_margin": 2.0}
    g = make_games(
        [
            ("2022-23", "2022-11-01", "AAA", "BBB", 100, 90),
            ("2022-23", "2022-11-02", "AAA", "BBB", 100, 90, 100.0, True),
        ]
    )
    d = g.assign(prior_home=0.0, prior_away=0.0, prior_diff=0.0, home_flag=(~g.is_neutral).astype(float))
    p = preseason.predict(params, d)
    assert p.pred_margin.tolist() == [2.0, 0.0]
    assert p.pred_p_home.iloc[1] == pytest.approx(0.5) and p.const_p_home.iloc[1] == 0.5
