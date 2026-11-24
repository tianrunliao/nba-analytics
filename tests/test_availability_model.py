"""Availability model: reason categories, causal player-game features, availability model, lineup
contribution and return-time tables."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_games

from nba_sos.models import availability as AM


@pytest.mark.parametrize(
    "reason,cat",
    [
        ("Injury/Illness - Left Ankle; Sprain", "injury"),
        ("G League - Two-Way", "g_league"),
        ("Health and Safety Protocols", "health_protocol"),
        ("Injury/Illness-LeftKnee;InjuryManagement", "injury_management"),
        ("Rest", "rest"),
        ("Not With Team", "not_with_team"),
        ("League Suspension", "suspension"),
        ("Return to Competition Reconditioning", "reconditioning"),
        ("", "unknown"),
    ],
)
def test_reason_category(reason, cat):
    assert AM.reason_category(reason) == cat


def _season(n_games=8):
    rows = [("2022-23", f"2022-11-{d:02d}", "AAA", "BBB", 100 + d, 100) for d in range(1, n_games + 1)]
    return make_games(rows)


def _pg(g, played_by_game):
    """player 10 on AAA; minutes per game index (0 = did not play)."""
    recs = []
    for gid, d, m in zip(g.game_id, g.game_date, played_by_game):
        if m > 0:
            recs.append(
                {
                    "season": "2022-23",
                    "game_id": gid,
                    "game_date": d,
                    "player_id": 10,
                    "player_name": "P",
                    "player_key": "p",
                    "team_id": 1,
                    "minutes": float(m),
                    "plus_minus": 0.0,
                }
            )
    # one full-time teammate so the roster exists on every date
    for gid, d in zip(g.game_id, g.game_date):
        recs.append(
            {
                "season": "2022-23",
                "game_id": gid,
                "game_date": d,
                "player_id": 11,
                "player_name": "Q",
                "player_key": "q",
                "team_id": 1,
                "minutes": 30.0,
                "plus_minus": 0.0,
            }
        )
    d = pd.DataFrame(recs)
    d["player_id"], d["team_id"] = d.player_id.astype("Int64"), d.team_id.astype("Int64")
    return d


PV = pd.DataFrame({"season": ["2021-22"], "player_id": [10], "lpm": [4.0], "min_total": [2000.0], "gp": [70]})


def test_player_game_frame_causal_minutes_and_streak():
    g = _season(8)
    pg = _pg(g, [30, 20, 0, 0, 40, 10, 0, 25])
    f = AM.player_game_frame(g, pg, PV)
    p = f[f.player_id == 10].sort_values("game_date").reset_index(drop=True)
    assert p.streak_before.tolist() == [0, 0, 0, 1, 2, 0, 0, 1]
    assert np.isnan(p.mpg_last10[0])
    assert p.mpg_last10[1] == 30 and p.mpg_last10[2] == 25 and p.mpg_last10[3] == 25  # missed games carry forward
    assert p.mpg_last10[5] == pytest.approx(30)  # (30+20+40)/3
    assert p.exp_mp[0] == pytest.approx(2000 / 70)  # falls back to last season
    assert p.V[0] == pytest.approx(4.0 * 2000 / 2500)
    # perturbing a future game's minutes leaves earlier features unchanged
    f2 = AM.player_game_frame(g, _pg(g, [30, 20, 0, 0, 40, 10, 0, 5]), PV)
    p2 = f2[f2.player_id == 10].sort_values("game_date").reset_index(drop=True)
    pd.testing.assert_series_equal(p.mpg_last10[:7], p2.mpg_last10[:7])


def test_availability_model_learns_status_effects():
    rng = np.random.default_rng(0)
    n = 6000
    x = pd.DataFrame(
        {
            "status": rng.choice(["Out", "Questionable", "not_listed"], n),
            "rotation": rng.random(n) < 0.5,
            "reason_cat": "injury",
            "streak_bin": rng.choice(AM.STREAK_LABELS, n),
            "gp_before": 10,
        }
    )
    p_true = np.where(
        x.status == "Out", 0.01, np.where(x.status == "Questionable", np.where(x.rotation, 0.7, 0.35), 0.97)
    )
    x["y_played"] = (rng.random(n) < p_true).astype(float)
    m = AM.fit_availability(x)
    q = AM.predict_availability(m, x)
    for st, rot, target in [
        ("Out", True, 0.01),
        ("Questionable", True, 0.7),
        ("Questionable", False, 0.35),
        ("not_listed", False, 0.97),
    ]:
        sel = (x.status == st) & (x.rotation == rot)
        assert q[sel.to_numpy()].mean() == pytest.approx(target, abs=0.06)


def test_lineup_recent_uses_only_earlier_games():
    g = _season(5)
    pg = _pg(g, [30, 0, 0, 30, 30])
    x = AM.player_game_frame(g, pg, PV)
    q = np.ones(len(x))
    lf = AM.lineup_features(x, q, g, tau_days=1e9).sort_values("date").reset_index(drop=True)
    aaa = lf[lf.team_id == 1].reset_index(drop=True)
    assert np.isnan(aaa.Mv_recent[0])
    assert aaa.Mv_recent[1] == pytest.approx(0.0)  # nothing missing in game 1
    v_miss = (30 / 48) * 4.0 * 2000 / 2500
    assert aaa.Mv_actual[1] == pytest.approx(v_miss)
    assert aaa.Mv_recent[2] == pytest.approx(v_miss / 2)  # mean over games 1-2
    assert aaa.Mv_today.max() == 0.0  # q = 1: nobody expected out


def test_return_table_hazard_and_cumulative():
    f = pd.DataFrame(
        {
            "rotation": True,
            "gp_before": 10,
            "streak_before": [1] * 10 + [2] * 10,
            "y_played": [1] * 5 + [0] * 5 + [1] * 2 + [0] * 8,
        }
    )
    t = AM.return_table(f).set_index("streak_before")
    assert t.loc[1, "p_return_next"] == 0.5 and t.loc[2, "p_return_next"] == 0.2
    assert 0.5 <= t.loc[1, "p_back_within_5"] <= 1.0
