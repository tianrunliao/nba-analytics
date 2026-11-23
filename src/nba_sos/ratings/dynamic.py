"""Pre-game dynamic team strength (strictly causal, per 100 possessions).

EWMA net rating (plan 5.1): for team i before date d,
    R_i = sum_k exp(-(d - t_k)/tau) * y_ik / sum_k exp(-(d - t_k)/tau)
over all of the team's earlier games (across seasons; the off-season gap decays old
games naturally). y = the team's per-100 margin in that game.

Dynamic opponent-adjusted strength (plan 5.2): before each game date d of season s, a
time-decayed ridge regression on season-s games before d,

    min sum_g w_g (y_g - S_h + S_a - HCA)^2 + alpha * sum_j (S_j - prior_j)^2,
    w_g = exp(-(d - t_g)/tau),

with prior_j = rho * (team j's final adjusted strength in season s-1) and HCA fixed at the
previous seasons' estimate. Also returns the posterior sd of each S_j (uncertainty).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def team_long(games: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame({"game_id": games.game_id, "season": games.season, "date": pd.to_datetime(games.game_date),
                      "team_id": games.home_team_id.astype(int), "y": games.net_rating_margin.astype(float)})
    a = h.assign(team_id=games.away_team_id.astype(int).to_numpy(), y=-games.net_rating_margin.astype(float).to_numpy())
    return pd.concat([h, a], ignore_index=True).sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)


def ewma_pregame(games: pd.DataFrame, tau_days: float) -> pd.DataFrame:
    """game_id, team_id, ewma_net (NaN for a team's first ever game in the data)."""
    t = team_long(games)
    out = []
    for tid, g in t.groupby("team_id"):
        days = (g.date - g.date.min()).dt.days.to_numpy(float)
        y = g.y.to_numpy(float)
        num = den = 0.0
        last = None
        vals = np.empty(len(g))
        for i in range(len(g)):
            if last is not None:
                decay = np.exp(-(days[i] - last) / tau_days)
                num, den = num * decay, den * decay
            vals[i] = num / den if den > 0 else np.nan
            num, den = num + y[i], den + 1.0
            last = days[i]
        out.append(pd.DataFrame({"game_id": g.game_id.to_numpy(), "team_id": tid, "ewma_net": vals}))
    return pd.concat(out, ignore_index=True)


def season_final_strength(games: pd.DataFrame, alpha: float = 10.0) -> pd.DataFrame:
    """Full-season schedule-adjusted strength (ridge, per season) used as next season's prior."""
    out = []
    for s, g in games.groupby("season"):
        teams = sorted(set(g.home_team_id.astype(int)) | set(g.away_team_id.astype(int)))
        S, _ = _fit(g, teams, np.ones(len(g)), np.zeros(len(teams)), alpha, hca=None)
        out.append(pd.DataFrame({"season": s, "team_id": teams, "final_strength": S[: len(teams)]}))
    return pd.concat(out, ignore_index=True)


def _fit(g: pd.DataFrame, teams: list[int], w: np.ndarray, prior: np.ndarray, alpha: float, hca: float | None):
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(g), len(teams)
    X = np.zeros((n, k + (1 if hca is None else 0)))
    X[np.arange(n), g.home_team_id.astype(int).map(idx).to_numpy()] = 1.0
    X[np.arange(n), g.away_team_id.astype(int).map(idx).to_numpy()] = -1.0
    home = (~g.is_neutral).to_numpy(float)
    y = g.net_rating_margin.to_numpy(float)
    if hca is None:
        X[:, k] = home
    else:
        y = y - hca * home
    pen = np.zeros(X.shape[1])
    pen[:k] = alpha
    pr = np.zeros(X.shape[1])
    pr[:k] = prior
    XtW = X.T * w
    A = XtW @ X + np.diag(pen)
    S = np.linalg.solve(A, XtW @ y + pen * pr)
    return S, A


def _season_setup(games: pd.DataFrame, final: pd.DataFrame, seasons: list[str], i: int, rho: float):
    """Teams, prior and home-court value for season ``seasons[i]`` from earlier seasons only."""
    gs = games[games.season == seasons[i]]
    teams = sorted(set(gs.home_team_id.astype(int)) | set(gs.away_team_id.astype(int)))
    if i == 0:
        # first season of the data: no earlier games, so no prior and no home-court
        # estimate (using this season's own games would leak future results)
        return teams, np.zeros(len(teams)), 0.0
    prev = final[final.season == seasons[i - 1]].set_index("team_id").final_strength
    prior = rho * prev.reindex(teams).fillna(0.0).to_numpy(float)
    past = games[games.season.isin(seasons[:i])]
    return teams, prior, float(past[~past.is_neutral].net_rating_margin.mean())


def _fit_on_date(before: pd.DataFrame, d: pd.Timestamp, teams: list[int], prior: np.ndarray, alpha: float,
                 tau_days: float, hca: float, noise_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Strength and posterior sd of every team from the season's games before date d."""
    if len(before):
        w = np.exp(-(d - before.date).dt.days.to_numpy(float) / tau_days)
        S, A = _fit(before, teams, w, prior, alpha, hca)
        return S[: len(teams)], (noise_sd * np.sqrt(np.diag(np.linalg.inv(A))))[: len(teams)]
    return prior.copy(), noise_sd / np.sqrt(alpha) * np.ones(len(teams))


def dynamic_pregame(games: pd.DataFrame, tau_days: float, alpha: float, rho: float,
                    noise_sd: float = 13.0) -> pd.DataFrame:
    """game_id, team_id, dyn_strength, dyn_sd for every game (pre-game)."""
    final = season_final_strength(games)
    seasons = sorted(games.season.unique())
    rows = []
    for i, s in enumerate(seasons):
        gs = games[games.season == s].copy()
        gs["date"] = pd.to_datetime(gs.game_date)
        teams, prior, hca = _season_setup(games, final, seasons, i, rho)
        for d, today in gs.groupby("date"):
            S, sd = _fit_on_date(gs[gs.date < d], d, teams, prior, alpha, tau_days, hca, noise_sd)
            sm = dict(zip(teams, S))
            sdm = dict(zip(teams, sd))
            for r in today.itertuples(index=False):
                for tid in (int(r.home_team_id), int(r.away_team_id)):
                    rows.append((r.game_id, tid, sm[tid], sdm[tid]))
    return pd.DataFrame(rows, columns=["game_id", "team_id", "dyn_strength", "dyn_sd"])


def strengths_asof(games: pd.DataFrame, season: str, cutoffs: list, tau_days: float, alpha: float, rho: float,
                   noise_sd: float = 13.0, prior_fn=None) -> pd.DataFrame:
    """Strength of every team of ``season`` as of each cutoff date (games strictly before it),
    with the same prior / home-court rules as :func:`dynamic_pregame`. ``games`` holds
    completed games only (this season to date and earlier seasons).
    ``prior_fn(cutoff, teams, final_prev) -> pd.Series`` replaces rho * last season's final
    strength as the ridge prior (e.g. the roster prior); it must use information before the
    cutoff only. Returns cutoff, team_id, S, sd, hca."""
    final = season_final_strength(games[games.season != season])
    seasons = sorted(games.season.unique())
    i = seasons.index(season)
    teams, prior, hca = _season_setup(games, final, seasons, i, rho)
    prev_final = final[final.season == seasons[i - 1]].set_index("team_id").final_strength if i else pd.Series(dtype=float)
    gs = games[games.season == season].copy()
    gs["date"] = pd.to_datetime(gs.game_date)
    out = []
    for c in cutoffs:
        d = pd.Timestamp(c)
        if prior_fn is not None:
            prior = prior_fn(d, teams, prev_final).reindex(teams).to_numpy(float)
        S, sd = _fit_on_date(gs[gs.date < d], d, teams, prior, alpha, tau_days, hca, noise_sd)
        out.append(pd.DataFrame({"cutoff": d, "team_id": teams, "S": S, "sd": sd, "hca": hca}))
    return pd.concat(out, ignore_index=True)
