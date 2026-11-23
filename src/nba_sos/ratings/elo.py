"""Margin-of-victory Elo (pre-game ratings, strictly causal).

Update after each game (FiveThirtyEight-style margin multiplier):

    p_home = 1 / (1 + 10^(-(R_h - R_a + HCA*[not neutral]) / 400))
    mult   = (|MOV| + 3)^0.8 / (7.5 + 0.006 * |elo_diff_winner|)
    R_h   += K * mult * (result - p_home);  R_a -= same

Between seasons ratings regress toward the mean: R <- (1 - carry) * 1500 + carry * R.
K, HCA and carry are chosen by log loss on training seasons only (``tune``).
Ratings stored for a game are those BEFORE it is played.
"""
from __future__ import annotations

import itertools

import pandas as pd

from ..evaluation import metrics as M

GRID = {"k": (15.0, 20.0, 25.0, 30.0, 40.0), "hca": (50.0, 75.0, 100.0), "carry": (0.5, 0.65, 0.8)}


def run_elo(games: pd.DataFrame, k: float, hca: float, carry: float) -> pd.DataFrame:
    """``games`` in any order; returns game_id, home_elo, away_elo, p_home_elo (pre-game)."""
    g = games.sort_values(["tipoff_utc", "game_id"])
    rating: dict[int, float] = {}
    season_prev = None
    rows = []
    for r in g.itertuples(index=False):
        if r.season != season_prev:
            rating = {t: (1 - carry) * 1500 + carry * v for t, v in rating.items()}
            season_prev = r.season
        h, a = int(r.home_team_id), int(r.away_team_id)
        rh, ra = rating.get(h, 1500.0), rating.get(a, 1500.0)
        adv = 0.0 if r.is_neutral else hca
        p = 1 / (1 + 10 ** (-(rh - ra + adv) / 400))
        rows.append((r.game_id, rh, ra, p))
        mov = float(r.home_score) - float(r.away_score)
        res = 1.0 if mov > 0 else 0.0
        diff_w = (rh + adv - ra) if mov > 0 else (ra - rh - adv)
        mult = (abs(mov) + 3) ** 0.8 / (7.5 + 0.006 * diff_w)
        delta = k * mult * (res - p)
        rating[h], rating[a] = rh + delta, ra - delta
    return pd.DataFrame(rows, columns=["game_id", "home_elo", "away_elo", "p_home_elo"])


def tune(games: pd.DataFrame, train_seasons: list[str], burn_in: list[str]) -> dict:
    """Grid search on train seasons (excluding the burn-in seasons from scoring)."""
    sub = games[games.season.isin(train_seasons)]
    scored = sub[~sub.season.isin(burn_in)]
    best = None
    for k, hca, carry in itertools.product(GRID["k"], GRID["hca"], GRID["carry"]):
        e = run_elo(sub, k, hca, carry).set_index("game_id").loc[scored.game_id]
        ll = M.log_loss(scored.home_win.astype(float).to_numpy(), e.p_home_elo.to_numpy())
        if best is None or ll < best[0]:
            best = (ll, {"k": k, "hca": hca, "carry": carry})
    return {**best[1], "train_log_loss": best[0]}
