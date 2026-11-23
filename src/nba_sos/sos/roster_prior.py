"""Roster-based strength prior for the dynamic model (replaces rho * last season's strength).

Roster value of team j at a cutoff:

    R_j = sum_p V_p * m_p,   m_p = 5 * f_p / sum_q f_q

over the players whose most recent appearance before the cutoff was for team j (on
opening night: the players who appear in the team's first game), V = previous-season
player value (LPM, minutes-shrunk; 0 without one) and f = expected minutes / 48. Trades
and signings therefore move R, unlike last season's team strength.

Prior = a * (team's final strength last season) + b * R_j + c, with (a, b, c) fit on the
team-seasons of the training seasons only: target = the season's final schedule-adjusted
strength, R measured on opening night. Grouped k-fold over seasons compares the blend
with the old rho * last-season rule out of fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import season_start_year
from ..evaluation.metrics import season_folds


def roster_value(pgf_season: pd.DataFrame, cutoff: pd.Timestamp | None) -> pd.Series:
    """team_id -> R. ``cutoff`` None = opening night (each team's first game)."""
    if cutoff is None:
        first = pgf_season.groupby("team_id").game_date.transform("min")
        x = pgf_season[(pgf_season.game_date == first) & pgf_season.played]
    else:
        b = pgf_season[pgf_season.game_date < cutoff]
        if b.empty:
            return roster_value(pgf_season, None)
        x = b.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
    f = x.exp_mp.astype(float).fillna(0.0) / 48.0
    # opening night: exp_mp is empty (no games yet) -> previous season mpg
    f = f.where(f > 0, x.mpg_prev.astype(float).fillna(0.0) / 48.0)
    x = x.assign(f=f.to_numpy(), V=x.V.astype(float).to_numpy())
    tot = x.groupby("team_id").f.transform("sum")
    x = x.assign(m=5 * x.f / tot.replace(0, np.nan))
    return x.assign(r=x.m.fillna(0) * x.V).groupby(x.team_id.astype(int)).r.sum()


def training_table(pgf: pd.DataFrame, final: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Per team-season: final strength, last season's final strength, opening-night R."""
    rows = []
    fin = final.set_index(["season", "team_id"]).final_strength
    for s in seasons:
        prev = f"{season_start_year(s) - 1}-{str(season_start_year(s))[-2:]}"
        R = roster_value(pgf[pgf.season == s], None)
        for t, r in R.items():
            if (s, t) in fin.index:
                rows.append({"season": s, "team_id": t, "y": fin[(s, t)], "prev": fin.get((prev, t), np.nan), "R": r})
    return pd.DataFrame(rows).dropna()


def _fit(d: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([d.prev, d.R, np.ones(len(d))])
    b, *_ = np.linalg.lstsq(X, d.y.to_numpy(float), rcond=None)
    return b


def fit_prior(table: pd.DataFrame, rho: float, k: int = 5) -> dict:
    """Coefficients from all rows; out-of-fold MSE of the blend vs rho * last season."""
    oof, base = [], []
    for tr, va in season_folds(sorted(table.season.unique()), k):
        b = _fit(table[table.season.isin(tr)])
        v = table[table.season.isin(va)]
        oof.append(v.y - (b[0] * v.prev + b[1] * v.R + b[2]))
        base.append(v.y - rho * v.prev)
    oof, base = pd.concat(oof), pd.concat(base)
    b = _fit(table)
    return {"a_prev": float(b[0]), "b_roster": float(b[1]), "c": float(b[2]), "n": int(len(table)),
            "oof_mse": float((oof ** 2).mean()), "oof_mse_rho_prev": float((base ** 2).mean()), "rho": rho}


def prior_at(pgf_season: pd.DataFrame, final_prev: pd.Series, cutoff: pd.Timestamp | None, coef: dict,
             teams: list[int]) -> pd.Series:
    R = roster_value(pgf_season, cutoff).reindex(teams).fillna(0.0)
    prev = final_prev.reindex(teams).fillna(0.0)
    return coef["a_prev"] * prev + coef["b_roster"] * R + coef["c"]
