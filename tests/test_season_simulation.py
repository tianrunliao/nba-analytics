"""Season simulation: schedule difficulty, joint league simulation, horizon calibration, tie-breaks,
NBA Cup, availability scenarios, roster prior, k-fold selection, no look-ahead."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_games

from nba_sos.evaluation.metrics import season_folds
from nba_sos.ratings import dynamic as DY
from nba_sos.sos import availability as AV
from nba_sos.sos import cup as CUP
from nba_sos.sos import projection as PJ
from nba_sos.sos import roster_prior as RP
from nba_sos.sos.standings import Season

PRE = {"hca": 2.0, "c": {"b2b": -2.0, "g4": -0.5, "km": -0.1, "tz_abs": -0.3}, "sigma": 13.0}
HZ = {"b0": 1.0, "r_per_day": 0.0, "kappa": 0.2, "q_per_day": 0.1, "sigma_game": 12.0}
CUT = pd.Timestamp("2026-01-01")


def _league(n_teams=6, rounds=4, seed=0):
    """Every pair meets ``rounds`` times, alternating venues, one game per day."""
    rng = np.random.default_rng(seed)
    rows, day = [], 0
    for r in range(rounds):
        for i in range(1, n_teams + 1):
            for j in range(i + 1, n_teams + 1):
                h, a = (i, j) if r % 2 == 0 else (j, i)
                rows.append(
                    {
                        "game_id": f"g{len(rows)}",
                        "season": "x",
                        "date": CUT + pd.Timedelta(int(day), unit="D"),
                        "home_team_id": h,
                        "away_team_id": a,
                        "venue": 1.0,
                        **{f"{k}_{s}": float(rng.integers(0, 2)) for k in PJ.LOAD for s in ("home", "away")},
                    }
                )
                day += 1
    return pd.DataFrame(rows)


def _strengths(n_teams=6):
    return pd.DataFrame({"team_id": range(1, n_teams + 1), "S": np.linspace(-6, 6, n_teams), "sd": 3.0})


# ------------------------------------------------------------------ k-fold
def test_season_folds_hold_out_whole_seasons_once():
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2008, 2024)]
    folds = season_folds(seasons, 5)
    held = [s for _, va in folds for s in va]
    assert sorted(held) == sorted(seasons)  # every season validated exactly once
    for tr, va in folds:
        assert not set(tr) & set(va) and set(tr) | set(va) == set(seasons)


def test_load_selection_keeps_only_features_that_help_out_of_fold():
    rng = np.random.default_rng(2)
    rows = []
    for s in range(8):
        n = 1500
        d = pd.DataFrame(
            {
                "season": f"{2010 + s}-{str(2011 + s)[-2:]}",
                "dyn_diff": rng.normal(0, 5, n),
                "venue": 1.0,
                **{f"{k}_{sd}": rng.integers(0, 2, n).astype(float) for k in PJ.LOAD for sd in ("home", "away")},
            }
        )
        d["y"] = d.dyn_diff + 2 - 2.5 * (d.b2b_home - d.b2b_away) + rng.normal(0, 12, n)
        rows.append(d)
    sel = PJ.select_load(pd.concat(rows, ignore_index=True), 4)
    assert sel["features"] == ["b2b"]


# ------------------------------------------------------------------ schedule difficulty / simulation
def test_shapley_parts_add_up_and_average_schedule_is_neutral():
    g = _league()
    out = PJ.schedule_difficulty(PJ.team_games(g), _strengths(), PRE, HZ, CUT)
    parts = out[[f"part_{k}" for k in PJ.COMPONENTS]].sum(axis=1)
    np.testing.assert_allclose(parts, out.d_sos, atol=1e-10)
    flat = g.assign(venue=0.0, **{f"{k}_{s}": 0.0 for k in PJ.LOAD for s in ("home", "away")})
    z = PJ.schedule_difficulty(PJ.team_games(flat), _strengths().assign(S=0.0), PRE, HZ, CUT)
    np.testing.assert_allclose(z.d_sos, 0.0, atol=1e-12)
    assert out.sort_values("opp_sos").team_id.iloc[-1] == 1  # weakest team faces the strongest


def test_simulation_is_joint_with_one_winner_per_game():
    g = _league()
    sim = PJ.simulate(g, _strengths(), PRE, HZ, CUT, n_paths=500, seed=1)
    assert (sim["wins"].sum(axis=1) == len(g)).all()
    assert np.all(np.diff(sim["wins"].mean(axis=0)) > 0)


def test_horizon_moments_recover_team_level_error():
    rng = np.random.default_rng(5)
    kappa, q, sg = 0.5, 0.2, 12.0
    rows = []
    # cells differ in posterior sd and remaining schedule length, as real cutoffs do
    # (identical cells would leave kappa and q unidentified: their moment columns coincide)
    for c in range(90):
        n, sd = 30, [2.0, 4.0, 6.0][c % 3]
        u = rng.normal(0, np.sqrt(kappa) * sd, n)
        for k in range([8, 20, 40][(c // 3) % 3]):
            if k:
                u = u + rng.normal(0, np.sqrt(q * 3), n)
            perm = rng.permutation(n)
            for a, b in zip(perm[0::2], perm[1::2]):
                rows.append(
                    {
                        "cutoff": c,
                        "home_team_id": a,
                        "away_team_id": b,
                        "h": float(k * 3),
                        "venue": 0.0,
                        "S_home": 0.0,
                        "S_away": 0.0,
                        "sd_home": sd,
                        "sd_away": sd,
                        "sd2": 2 * sd**2,
                        "y": u[a] - u[b] + rng.normal(0, sg),
                        **{f"{k2}_{s}": 0.0 for k2 in PRE["c"] for s in ("home", "away")},
                    }
                )
    fit = PJ.fit_horizon(pd.DataFrame(rows), PRE)
    assert fit["kappa"] == pytest.approx(kappa, rel=0.3)
    assert fit["q_per_day"] == pytest.approx(q, rel=0.3)
    assert fit["sigma_game"] == pytest.approx(sg, rel=0.05)


def test_strengths_asof_ignore_games_on_or_after_cutoff_and_match_pregame():
    rows = [
        ("2021-22", f"2021-11-{d:02d}", h, a, 100 + 3 * d % 7, 100)
        for d, (h, a) in zip(range(1, 13), [("AAA", "BBB"), ("CCC", "DDD"), ("AAA", "CCC"), ("BBB", "DDD")] * 3)
    ]
    rows += [
        ("2022-23", f"2022-11-{d:02d}", h, a, 100 + (5 * d) % 9, 100)
        for d, (h, a) in zip(range(1, 13), [("BBB", "AAA"), ("DDD", "CCC"), ("CCC", "AAA"), ("DDD", "BBB")] * 3)
    ]
    g = make_games(rows)
    cut = pd.Timestamp("2022-11-07")
    a = DY.strengths_asof(g, "2022-23", [cut], 60, 5, 0.7)
    g2 = g.copy()
    g2.loc[(g2.season == "2022-23") & (pd.to_datetime(g2.game_date) >= cut), "net_rating_margin"] = 50.0
    pd.testing.assert_frame_equal(a, DY.strengths_asof(g2, "2022-23", [cut], 60, 5, 0.7))
    pre = DY.dynamic_pregame(g, 60, 5, 0.7)
    on_cut = g[(g.season == "2022-23") & (pd.to_datetime(g.game_date) == cut)].game_id
    x = pre[pre.game_id.isin(on_cut)].merge(a, on="team_id")
    np.testing.assert_allclose(x.dyn_strength, x.S)
    b = DY.strengths_asof(g, "2022-23", [cut], 60, 5, 0.7, prior_fn=lambda c, t, prev: pd.Series(0.0, index=t))
    assert not np.allclose(a.S, b.S)


# ------------------------------------------------------------------ tie-breaks
def _season(wins, W, pdiff, conf, div, seed=0):
    return Season(wins, W, pdiff, conf, div, np.random.default_rng(seed))


def test_two_team_tie_head_to_head_then_division_winner():
    W = np.zeros((4, 4))
    W[1, 0], W[0, 1] = 3, 1  # 1 won the season series
    s = _season([50, 50, 30, 20], W, [0, 0, 0, 0], ["E"] * 4, ["A", "B", "A", "B"])
    assert s.conference_order("E")[:2] == [1, 0]
    W2 = np.zeros((4, 4))
    W2[1, 0], W2[0, 1] = 2, 2  # split series: division winner next
    s2 = _season([50, 50, 30, 55], W2, [0, 0, 0, 0], ["E"] * 4, ["A", "B", "A", "B"])
    o = s2.conference_order("E")
    assert o.index(0) < o.index(1)  # 0 leads division A, 1 trails 3 in B


def test_multi_team_tie_partial_break_restarts_with_two_team_rules():
    # 0, 1, 2 tied: 2 wins division B and goes first; 0 and 1 (both division A) restart
    # with the two-team list, where head-to-head (0 beat 1 three times) decides
    W = np.zeros((4, 4))
    W[0, 1], W[1, 0] = 3, 1
    s = _season([40, 40, 40, 39], W, [-5, 5, 0, 0], ["E"] * 4, ["A", "A", "B", "B"])
    assert s.conference_order("E")[:3] == [2, 0, 1]


def test_point_differential_breaks_otherwise_identical_teams():
    s = _season([40, 40], np.zeros((2, 2)), [10.0, -10.0], ["E", "E"], ["A", "A"])
    assert s.conference_order("E") == [0, 1]


# ------------------------------------------------------------------ NBA Cup
def _cup_plan():
    groups, rows = {}, []
    tid = 1
    for conf in ("East", "West"):
        for gname in "ABC":
            ts = list(range(tid, tid + 5))
            tid += 5
            groups[f"{conf} Group {gname}"] = ts
            for i in range(5):
                for j in range(i + 1, 5):
                    rows.append(
                        {
                            "game_id": f"c{len(rows)}",
                            "home_team_id": ts[i],
                            "away_team_id": ts[j],
                            "group": f"{conf} Group {gname}",
                        }
                    )
    return CUP.CupPlan(
        groups=groups,
        group_games=pd.DataFrame(rows),
        qf_date=CUT + pd.Timedelta(10, unit="D"),
        sf_date=CUT + pd.Timedelta(13, unit="D"),
        extra_dates=[CUT + pd.Timedelta(11, unit="D"), CUT + pd.Timedelta(14, unit="D")],
        sf_home_court=True,
        prev_wins=pd.Series(dtype=float),
        conference={k: k.split()[0] for k in groups},
    )


def test_group_tables_use_head_to_head_then_point_differential():
    plan = _cup_plan()
    gg = plan.group_games
    hw = np.zeros(len(gg), bool)
    m = np.full(len(gg), -1.0)
    # East Group A (teams 1-5): 1 and 2 both 3-1, 1 beat 2
    res = {(1, 2): True, (1, 3): True, (1, 4): True, (1, 5): False, (2, 3): True, (2, 4): True, (2, 5): True}
    for i, r in gg[gg.group == "East Group A"].iterrows():
        win = res.get((r.home_team_id, r.away_team_id), r.home_team_id < r.away_team_id)
        hw[i], m[i] = win, 5.0 if win else -5.0
    t = CUP.group_tables(plan, hw, m, np.random.default_rng(0))["East Group A"]
    assert t["order"][:2] == [1, 2]


def test_cup_knockout_and_extra_games_give_every_team_two_more_games():
    plan = _cup_plan()
    st = pd.DataFrame({"team_id": range(1, 31), "S": np.linspace(-5, 5, 30), "sd": 2.0})
    gg = plan.group_games.assign(
        date=CUT + pd.Timedelta(1, unit="D"), venue=1.0, **{f"{k}_{s}": 0.0 for k in PRE["c"] for s in ("home", "away")}
    )
    sim = PJ.simulate(gg, st, PRE, HZ, CUT, n_paths=200, seed=3, cup=plan)
    out = sim["cup"]
    assert (out["qf"].sum(axis=1) == 8).all() and (out["sf"].sum(axis=1) == 4).all()
    assert (out["final"].sum(axis=1) == 2).all()
    # every team: 4 group games + 2 counting games (QF + SF, QF + loser game, or 2 extra)
    assert (sim["wins"].sum(axis=1) == len(gg) + 30).all()


def test_pair_extra_has_no_self_or_repeated_pairs():
    pairs = CUP.pair_extra(list(range(11)), 2, np.random.default_rng(0))
    assert len(pairs) == 11
    assert all(a != b for a, b in pairs)
    assert len({tuple(sorted(p)) for p in pairs}) == 11
    assert (pd.Series([t for p in pairs for t in p]).value_counts() == 2).all()


# ------------------------------------------------------------------ availability / roster prior
def test_survival_and_sampled_returns_agree():
    h = np.full(AV.N_MAX + AV.K_MAX + 2, 0.3)
    s = AV.survival(h, 2)
    assert s[0] == 1.0 and s[1] == pytest.approx(0.7) and s[3] == pytest.approx(0.7**3)
    players = pd.DataFrame({"team_id": [1, 1], "player_id": [10, 11], "V": [4.0, 0.0], "f": [0.7, 0.6], "n": [2, 0]})
    teams = pd.DataFrame({"team_id": [1], "rec_v": [1.0], "rec_f": [0.5]})
    pi = np.linspace(0, 0.2, AV.K_MAX + 1)
    ex = AV.expected_change(players, teams, h, pi, [1])[1]
    assert ex["ret_v"][3] == pytest.approx(0.7**3 * 0.7 * 4.0)
    assert ex["new_f"][5] == pytest.approx(pi[5] * 0.6)
    sp = AV.sample_paths(players, teams, h, pi, [1], 20000, np.random.default_rng(0))[1]
    assert (sp["R"] >= 3).mean() == pytest.approx(0.7**3, abs=0.01)


def test_report_overlay_next_game_probability_and_mixture():
    h = np.full(AV.N_MAX + AV.K_MAX + 2, 0.3)
    pi = np.linspace(0, 0.2, AV.K_MAX + 1)
    players = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 2],
            "player_id": [10, 11, 12, 20],
            "V": [4.0, 2.0, 1.0, 3.0],
            "f": [0.7, 0.6, 0.5, 0.6],
            "n": [2, 0, 0, 3],
        }
    )
    teams = pd.DataFrame({"team_id": [1, 2], "rec_v": [0.0, 0.0], "rec_f": [0.0, 0.0]})
    avail = pd.DataFrame({"player_id": [11], "injury_status": ["Out"]})
    ts = pd.DataFrame({"team_id": [1], "snapshot_usable": [True]})
    table = pd.Series({"Out": 1.0, "not_listed": 0.03})
    over = AV.report_overlay(players, avail, ts, table).set_index("player_id")
    assert over.report_status[10] == "not_listed" and over.report_status[11] == "Out"
    assert np.isnan(over.p1[20])  # team 2 not covered
    base = AV.expected_change(players, teams, h, pi, [1, 2])
    ex = AV.expected_change(over.reset_index(), teams, h, pi, [1, 2])
    # 10: out 2 games, not listed -> back with prob 0.97; later games continue from n + 1 = 3
    # 11: healthy, listed Out -> misses game 1, then the hazards from one game missed
    # 12: healthy, not listed -> misses game 1 with p1 = 0.03 (then the hazards) and keeps
    #     new injuries from game 2 on, weighted by 1 - p1
    assert ex[1]["ret_v"][1] == pytest.approx(0.03 * 0.7 * 4.0 + 1.0 * 0.6 * 2.0 + 0.03 * 0.5 * 1.0)
    assert ex[1]["ret_v"][3] == pytest.approx(0.7**2 * (0.03 * 0.7 * 4.0 + 0.6 * 2.0 + 0.03 * 0.5 * 1.0))
    assert ex[1]["new_f"][1] == 0.0 and ex[1]["new_f"][5] == pytest.approx(0.97 * pi[5] * 0.5)
    for c in ("ret_v", "ret_f", "new_v", "new_f"):
        np.testing.assert_allclose(ex[2][c], base[2][c])  # uncovered team unchanged
    sp = AV.sample_paths(over.reset_index(), teams, h, pi, [1], 5000, np.random.default_rng(1))[1]
    assert (sp["R"][:, 1] >= 1).all()  # listed Out: misses game 1


def test_live_report_is_used_only_for_games_tipping_off_after_publication():
    from nba_sos.models import season_simulation as SIM

    tip = pd.to_datetime(["2026-11-24 23:00", "2026-11-24 18:00"], utc=True)
    games = pd.DataFrame(
        {
            "season": "2026-27",
            "game_id": ["g1", "g2"],
            "game_date": [pd.Timestamp("2026-11-24")] * 2,
            "tipoff_utc": tip,
            "home_team_id": [1, 3],
            "away_team_id": [2, 4],
            "home_team": ["AAA", "CCC"],
            "away_team": ["BBB", "DDD"],
        }
    )
    report = pd.DataFrame(
        {
            "game_date": ["2026-11-24"] * 2,
            "matchup": ["BBB@AAA", "DDD@CCC"],
            "team_tricode": ["AAA", "CCC"],
            "player": ["One,A", "Two,B"],
            "player_key": ["aone", "btwo"],
            "status": ["Out", "Out"],
            "reason": ["x", "x"],
            "not_submitted": [False, False],
            "status_valid": [True, True],
            "away_tricode": ["BBB", "DDD"],
            "home_tricode": ["AAA", "CCC"],
        }
    )
    pg = pd.DataFrame({"season": "2026-27", "team_id": [1, 3], "player_key": ["aone", "btwo"], "player_id": [7, 8]})
    rec = {"path": "injury_reports/x.pdf", "published_at_utc": "2026-11-24T20:00:00+00:00"}
    pa, ts = SIM.report_inputs_for_day(report, rec, games, pg)
    usable = ts.set_index(["game_id", "team_id"]).snapshot_usable
    assert usable[("g1", 1)] and not usable[("g2", 3)]  # g2 tipped before publication
    assert list(pa.player_id.dropna().astype(int)) == [7]


def test_state_at_uses_only_rotation_players_before_cutoff():
    d = pd.to_datetime(["2023-11-01", "2023-11-03", "2023-11-05"])
    pgf = pd.DataFrame(
        {
            "season": "2023-24",
            "game_id": np.repeat(["a", "b", "c"], 2),
            "game_date": np.repeat(d, 2),
            "player_id": [1, 2] * 3,
            "team_id": 7,
            "played": [True, True, False, True, False, False],
            "y_played": [1, 1, 0, 1, 0, 0],
            "streak_before": [0, 0, 0, 0, 1, 0],
            "gp_before": [3, 3, 4, 4, 4, 5],
            "V": [3.0, -1.0] * 3,
            "exp_mp": [30.0, 10.0] * 3,
            "rotation": [True, False] * 3,
            "minutes": 0.0,
            "mpg_prev": 30.0,
            "v_prev": 3.0,
        }
    )
    players, _ = AV.state_at(pgf, pd.Timestamp("2023-11-05"), 60)
    assert list(players.player_id) == [1] and players.n.iloc[0] == 1  # missed game b, c not yet played
    players2, _ = AV.state_at(pgf, pd.Timestamp("2023-11-06"), 60)
    assert players2.n.iloc[0] == 2


def test_roster_value_normalises_minutes_and_prior_fit_recovers_blend():
    pgf = pd.DataFrame(
        {
            "season": "s",
            "game_id": "g",
            "game_date": pd.Timestamp("2023-10-25"),
            "player_id": [1, 2, 3],
            "team_id": [5, 5, 5],
            "played": True,
            "exp_mp": [36.0, 24.0, 0.0],
            "mpg_prev": [36.0, 24.0, 12.0],
            "V": [2.0, 1.0, 0.0],
        }
    )
    R = RP.roster_value(pgf, None)
    f = np.array([36, 24, 12]) / 48
    assert R[5] == pytest.approx((5 * f / f.sum() * np.array([2.0, 1.0, 0.0])).sum())
    rng = np.random.default_rng(0)
    n = 600
    tab = pd.DataFrame(
        {
            "season": np.repeat([f"{2000 + i}-{str(2001 + i)[-2:]}" for i in range(10)], n // 10),
            "prev": rng.normal(0, 5, n),
            "R": rng.normal(0, 3, n),
        }
    )
    tab["y"] = 0.3 * tab.prev + 0.8 * tab.R - 0.5 + rng.normal(0, 1, n)
    c = RP.fit_prior(tab, 0.7)
    assert c["a_prev"] == pytest.approx(0.3, abs=0.05) and c["b_roster"] == pytest.approx(0.8, abs=0.05)
    assert c["oof_mse"] < c["oof_mse_rho_prev"]


def test_tiebreaks_reproduce_official_standings_since_2015_16():
    """Real data: rebuilt 1-15 conference order vs the official PlayoffRank. Before 2015-16
    division winners were guaranteed a top-4 seed (not modelled); the 2020-21 standings
    payload lists Memphis (play-in winner) 8th ahead of 39-33 Golden State."""
    from nba_sos.config import PROJECT_ROOT, Paths
    from nba_sos.ingestion import nba_stats
    from nba_sos.ingestion.raw_store import RawStore

    p = Paths(PROJECT_ROOT)
    if not (p.processed / "games.parquet").exists():
        pytest.skip("processed data not built")
    g = pd.read_parquet(p.processed / "games.parquet")
    store = RawStore(p.raw)
    wrong = []
    for s in sorted(x for x in g.season.unique() if x >= "2015-16"):
        sd = nba_stats.load(store, "standings", s)
        sd = sd.set_index(sd.TeamID.astype(int))
        teams = sorted(sd.index)
        pos = {t: i for i, t in enumerate(teams)}
        x = g[g.season == s]
        T = len(teams)
        W, pdiff, wins = np.zeros((T, T)), np.zeros(T), np.zeros(T)
        for h, a, hs, as_ in zip(
            x.home_team_id.astype(int),
            x.away_team_id.astype(int),
            x.home_score.astype(float),
            x.away_score.astype(float),
        ):
            i, j = pos[h], pos[a]
            w, l_ = (i, j) if hs > as_ else (j, i)
            W[w, l_] += 1
            wins[w] += 1
            pdiff[i] += hs - as_
            pdiff[j] -= hs - as_
        S = Season(
            wins,
            W,
            pdiff,
            sd.Conference.reindex(teams).to_numpy(),
            sd.Division.reindex(teams).to_numpy(),
            np.random.default_rng(0),
        )
        for c in ("East", "West"):
            for r, i in enumerate(S.conference_order(c), 1):
                if int(sd.PlayoffRank.loc[teams[i]]) != r:
                    wrong.append((s, teams[i]))
    assert {s for s, _ in wrong} <= {"2020-21"} and len(wrong) <= 2


def _line_fixture(tmp_path, seasons=("2018-19", "2019-20")):
    from nba_sos.config import Paths

    p = Paths(tmp_path)
    for d in (p.configs, p.processed, p.reports):
        d.mkdir(parents=True)
    codes = [f"T{i:02d}" for i in range(30)]
    pd.DataFrame(
        [{"season": s, "home_team": c, "home_team_id": 100 + i} for s in seasons for i, c in enumerate(codes)]
    ).to_parquet(p.processed / "games.parquet")
    rows = []
    for s in seasons:
        for i in range(30):
            for v, e in (("full", 40.0), ("v1", 45.0)):
                rows.append(
                    {
                        "season": s,
                        "team_id": 100 + i,
                        "variant": v,
                        "cutoff_games": 0,
                        "exp_wins": e,
                        "final_wins": 40 + (i % 5),
                    }
                )
    pd.DataFrame(rows).to_csv(p.reports / "season_sim_replays.csv", index=False)
    return p, codes


def test_line_check_compares_lines_with_replays_on_the_same_team_seasons(tmp_path):
    from nba_sos.models import season_simulation as SIM

    p, codes = _line_fixture(tmp_path)
    lines = [
        {"season": s, "team": c, "line": 40 + (i % 5) + 1.5, "source_url": "u"}
        for s in ("2018-19", "2019-20")
        for i, c in enumerate(codes)
    ]
    pd.DataFrame(lines).to_csv(p.configs / SIM.LINES_FILE, index=False)
    out = SIM.line_check(p).set_index(["scope", "variant"])
    assert out.loc[("pooled", "preseason_line"), "mae"] == pytest.approx(1.5)
    assert out.loc[("pooled", "full"), "mae"] == pytest.approx(2.0)  # |40 - (40 + i % 5)|
    assert out.loc[("pooled", "preseason_line"), "team_seasons"] == 30  # 2019-20 excluded
    assert "2019-20" not in out.index.get_level_values("scope")


def test_line_file_validation(tmp_path):
    from nba_sos.models import season_simulation as SIM

    p, codes = _line_fixture(tmp_path)
    pd.DataFrame(columns=SIM.LINE_COLUMNS).to_csv(p.configs / SIM.LINES_FILE, index=False)
    assert SIM.line_check(p) is None  # template only
    bad = [
        {"season": "2018-19", "team": "BRK" if i == 0 else c, "line": 41, "source_url": "u"}
        for i, c in enumerate(codes)
    ]
    pd.DataFrame(bad).to_csv(p.configs / SIM.LINES_FILE, index=False)
    with pytest.raises(ValueError, match="not NBA tricodes"):
        SIM.line_check(p)
    pd.DataFrame(bad[1:5]).to_csv(p.configs / SIM.LINES_FILE, index=False)
    with pytest.raises(ValueError, match="30 teams"):
        SIM.line_check(p)
