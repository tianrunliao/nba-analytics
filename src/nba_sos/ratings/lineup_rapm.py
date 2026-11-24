"""Lineup-level regularised adjusted plus-minus with a box-score prior ("LPM").

For every five-man lineup l of a team-season (regular season):

    y_l = PLUS_MINUS_l / MIN_l * 48 * 100 / pace_s      (per 100 possessions)
    y_l = c + sum_{p in l} r_p + e_l,    weight = MIN_l

    minimise  sum_l MIN_l (y_l - c - X_l r)^2 + lam * sum_p (r_p - prior_p)^2

prior_p = kappa * pv_box, the player's box-score value (``player_value``) in "sum of five
players = lineup net rating" units; kappa (how far to trust the box score) and ``lam``
are chosen jointly by cross-validation. Different lineups separate a player's effect from
his teammates' — the shared-credit problem of raw on/off. Opponent lineups are not in
the data; they average out over a season and are treated as noise. ``lam`` (in minutes)
is chosen per season by 5-fold cross-validation over lineups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDAS = (500.0, 1500.0, 5000.0, 15000.0, 50000.0)
PRIOR_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)  # prior = kappa * pv_box; kappa chosen by CV


def lineup_matrix(lu: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    ids = lu.GROUP_ID.str.strip("-").str.split("-").map(lambda x: [int(v) for v in x])
    players = sorted({p for row in ids for p in row})
    idx = {p: i for i, p in enumerate(players)}
    X = np.zeros((len(lu), len(players)))
    for r, row in enumerate(ids):
        X[r, [idx[p] for p in row]] = 1.0
    return X, players


def _solve(X, y, w, prior, lam):
    """Weighted ridge toward ``prior`` with an unpenalised intercept."""
    y2 = y - X @ prior
    Xc = np.column_stack([np.ones(len(X)), X])
    XtW = Xc.T * w
    pen = np.diag([0.0] + [lam] * X.shape[1])
    b = np.linalg.solve(XtW @ Xc + pen, XtW @ y2)
    return b[0], prior + b[1:]


def fit_season(lu: pd.DataFrame, pace: float, prior_by_player: pd.Series, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    lu = lu[lu.MIN > 0].reset_index(drop=True)
    X, players = lineup_matrix(lu)
    y = (lu.PLUS_MINUS / lu.MIN * 48 * 100 / pace).to_numpy(float)
    w = lu.MIN.to_numpy(float)
    box = prior_by_player.reindex(players).fillna(0.0).to_numpy(float)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 5, len(lu))
    cv = {}
    for kappa in PRIOR_SCALES:
        for lam in LAMBDAS:
            err = 0.0
            for k in range(5):
                tr, te = fold != k, fold == k
                c, r = _solve(X[tr], y[tr], w[tr], kappa * box, lam)
                err += float(np.sum(w[te] * (y[te] - c - X[te] @ r) ** 2))
            cv[(kappa, lam)] = err / w.sum()
    kappa, lam = min(cv, key=cv.get)
    prior = kappa * box
    c, r = _solve(X, y, w, prior, lam)
    c0, r0 = _solve(X, y, w, np.zeros_like(prior), lam)  # same penalty, no prior (pure RAPM-style)
    minutes = X.T @ w
    out = pd.DataFrame({"player_id": players, "lpm": r, "rapm_no_prior": r0, "prior": prior, "lineup_minutes": minutes})
    return out, {
        "lambda": lam,
        "kappa": kappa,
        "cv": {f"{a}|{b}": v for (a, b), v in cv.items()},
        "intercept": float(c),
        "n_lineups": int(len(lu)),
    }


def lineup_box_design(
    lu: pd.DataFrame, pace: float, z: pd.DataFrame, features: list[str], demean: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One season: x_l = sum of the five players' era-standardised features, y_l = net per
    100, both demeaned (minutes-weighted) within team-season; weights = minutes."""
    zc = [f"z_{f}" for f in features]
    lu = lu[lu.MIN > 0].reset_index(drop=True)
    season = lu.season.iloc[0]
    zz = z[z.season == season].set_index("player_id")[zc]
    ids = lu.GROUP_ID.str.strip("-").str.split("-").map(lambda x: [int(v) for v in x])
    X = np.vstack([zz.reindex(row).fillna(0.0).to_numpy().sum(axis=0) for row in ids])
    y = (lu.PLUS_MINUS / lu.MIN * 48 * 100 / pace).to_numpy(float)
    w = lu.MIN.to_numpy(float)
    if not demean:  # keep between-team variation; centre on the season only
        wm = w / w.sum()
        return X - wm @ X, y - wm @ y, w
    team = lu.TEAM_ID.to_numpy()
    for t in np.unique(team):
        m = team == t
        wm = w[m] / w[m].sum()
        X[m] -= wm @ X[m]
        y[m] -= wm @ y[m]
    return X, y, w


def fit_box_on_lineups(
    designs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], features: list[str], alpha: float = 1000.0
) -> dict:
    """Box-score weights identified from lineup variation WITHIN team-seasons: team quality
    and cross-team box-score accounting cannot drive them — only which box profiles make a
    team's own lineups better or worse. ``designs``: outputs of ``lineup_box_design``."""
    X = np.vstack([d[0] for d in designs])
    y = np.concatenate([d[1] for d in designs])
    w = np.concatenate([d[2] for d in designs])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + alpha * np.eye(X.shape[1]), XtW @ y)
    return {"beta": dict(zip(features, beta)), "n_lineups": int(len(y))}
