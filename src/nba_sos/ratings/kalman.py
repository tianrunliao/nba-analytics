"""Kalman-filter team strength (state-space model, per 100 possessions, strictly causal).

State: every team's strength S_j. Observation for a game on date d:

    y = S_home - S_away + HCA * [not neutral] + e,   e ~ N(0, obs_sd^2)

with HCA fixed per season at the mean non-neutral home margin of the earlier seasons in the
data (0 in the first season), as in :mod:`ratings.dynamic`. Between game dates of a season
each strength follows a random walk with variance ``q`` per day; at each season start
S <- carry * S and P <- carry^2 * P + p0 * M (regression to the mean plus new-season
uncertainty; the first season starts at S = 0, P = p0 * M). Games identify only strength
differences, so the noise is projected onto the sum-to-zero subspace (M = I - 11'/k):
strengths and their sd are relative to the league average. All games of a date are one
joint update, so every output for a date uses games of earlier dates only.

Tuning (:func:`tune`): the mean forecasts depend on q / obs_sd^2 and p0 / obs_sd^2 only, so
q, p0 and carry are chosen on a grid with obs_sd = ``OBS_SD0`` by the same squared error as
the dynamic model (``game_model._tune_dynamic``) on the tuning seasons; then all variances
are rescaled by c^2 = mean(innovation^2) / mean(predictive variance) on those seasons, so
the reported sd is calibrated while the means are unchanged.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from ..config import season_start_year

PARAMS = ("q", "p0", "carry", "obs_sd")
GRID = {"q": (0.02, 0.04, 0.08, 0.16), "p0": (5.0, 10.0, 20.0, 40.0), "carry": (0.5, 0.6, 0.7, 0.8, 0.9)}
OBS_SD0 = 13.0


def _filter(games: pd.DataFrame, q: float, p0: float, carry: float, obs_sd: float) -> tuple[list[int], dict]:
    """Run the filter over ``games`` (completed games, any seasons). Returns the team list and,
    per season: hca, the season-start state, and per game date the pre-game state of that
    date's games and the state after its update."""
    g = games.assign(date=pd.to_datetime(games.game_date)).sort_values(["date", "game_id"])
    teams = sorted(set(g.home_team_id.astype(int)) | set(g.away_team_id.astype(int)))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    M = np.eye(k) - 1.0 / k
    S, P = np.zeros(k), np.zeros((k, k))
    seasons = sorted(g.season.unique(), key=season_start_year)
    out = {}
    for i, s in enumerate(seasons):
        past = g[g.season.isin(seasons[:i])]
        hca = float(past[~past.is_neutral.astype(bool)].net_rating_margin.mean()) if i else 0.0
        S, P = carry * S, carry**2 * P + p0 * M
        gs = g[g.season == s]
        dates = gs.date.to_numpy()
        hi = gs.home_team_id.astype(int).map(idx).to_numpy()
        ai = gs.away_team_id.astype(int).map(idx).to_numpy()
        y = gs.net_rating_margin.to_numpy(float) - hca * (~gs.is_neutral.astype(bool)).to_numpy(float)
        gid = gs.game_id.to_numpy()
        rec: dict[str, Any] = {
            "hca": hca,
            "start_S": S.copy(),
            "start_var": np.diag(P).copy(),
            "dates": [],
            "post_S": [],
            "post_var": [],
            "pre": [],
        }
        uniq, first = np.unique(dates, return_index=True)
        bounds = list(first) + [len(dates)]
        last = None
        for j, d in enumerate(uniq):
            sl = slice(bounds[j], bounds[j + 1])
            if last is not None:
                P = P + q * float((d - last) / np.timedelta64(1, "D")) * M
            h, a = hi[sl], ai[sl]
            var = np.diag(P)
            rec["pre"].append((gid[sl], h, a, S[h].copy(), S[a].copy(), var[h].copy(), var[a].copy(), P[h, a].copy()))
            n = len(h)
            H = np.zeros((n, k))
            H[np.arange(n), h], H[np.arange(n), a] = 1.0, -1.0
            HP = H @ P
            K = np.linalg.solve(HP @ H.T + obs_sd**2 * np.eye(n), HP).T
            S = S + K @ (y[sl] - H @ S)
            P = P - K @ HP
            P = (P + P.T) / 2
            rec["dates"].append(pd.Timestamp(d))
            rec["post_S"].append(S.copy())
            rec["post_var"].append(np.diag(P).copy())
            last = d
        out[s] = rec
    return teams, out


def _pregame_frame(teams: list[int], states: dict) -> pd.DataFrame:
    """One row per game: ids, both pre-game strengths and variances, their covariance, hca."""
    rows = []
    t = np.asarray(teams)
    for s, rec in states.items():
        for gid, h, a, sh, sa, vh, va, cha in rec["pre"]:
            rows.append(
                pd.DataFrame(
                    {
                        "season": s,
                        "game_id": gid,
                        "home_team_id": t[h],
                        "away_team_id": t[a],
                        "S_home": sh,
                        "S_away": sa,
                        "var_home": vh,
                        "var_away": va,
                        "cov_ha": cha,
                        "hca": rec["hca"],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def kalman_pregame(games: pd.DataFrame, q: float, p0: float, carry: float, obs_sd: float) -> pd.DataFrame:
    """game_id, team_id, kal_strength, kal_sd for every game (pre-game)."""
    teams, states = _filter(games, q, p0, carry, obs_sd)
    f = _pregame_frame(teams, states)
    home = pd.DataFrame(
        {"game_id": f.game_id, "team_id": f.home_team_id, "kal_strength": f.S_home, "kal_sd": np.sqrt(f.var_home)}
    )
    away = pd.DataFrame(
        {"game_id": f.game_id, "team_id": f.away_team_id, "kal_strength": f.S_away, "kal_sd": np.sqrt(f.var_away)}
    )
    return pd.concat([home, away], ignore_index=True)


def filter_states(games: pd.DataFrame, params: dict) -> dict:
    """Filtered states of ``games`` for :func:`strengths_asof` (run once, query many cutoffs)."""
    teams, states = _filter(games, *(float(params[p]) for p in PARAMS))
    # per-team share of the projected random-walk variance: diag(q * M) = q * (1 - 1/k)
    return {"teams": teams, "q_team": float(params["q"]) * (1 - 1 / len(teams)), "seasons": states}


def strengths_asof(fs: dict, season: str, cutoffs: list) -> pd.DataFrame:
    """Strength of every team as of each cutoff of ``season`` (games strictly before the
    cutoff; random-walk variance added up to the cutoff). Same columns as
    ``dynamic.strengths_asof``: cutoff, team_id, S, sd, hca."""
    rec = fs["seasons"][season]
    dates = np.array(rec["dates"], dtype="datetime64[ns]")
    out = []
    for c in cutoffs:
        d = pd.Timestamp(c)
        j = int(np.searchsorted(dates, np.datetime64(d), side="left")) - 1
        if j < 0:
            S, var = rec["start_S"], rec["start_var"]
        else:
            S = rec["post_S"][j]
            var = rec["post_var"][j] + fs["q_team"] * float((d - rec["dates"][j]) / pd.Timedelta(1, unit="D"))
        out.append(pd.DataFrame({"cutoff": d, "team_id": fs["teams"], "S": S, "sd": np.sqrt(var), "hca": rec["hca"]}))
    return pd.concat(out, ignore_index=True)


def tune(all_games: pd.DataFrame, tune_seasons: list[str]) -> dict:
    """Grid search of q, p0, carry (obs_sd = OBS_SD0) by squared error on ``tune_seasons``,
    then variance calibration (module docstring). Uses seasons up to the last tuning season."""
    last = season_start_year(tune_seasons[-1])
    sub = all_games[all_games.season.map(season_start_year) <= last]
    tune_g = sub[sub.season.isin(tune_seasons)]
    hca_tune = float(tune_g[~tune_g.is_neutral.astype(bool)].net_rating_margin.mean())
    y = tune_g.set_index("game_id").net_rating_margin.astype(float)
    best = None
    for q, p0, carry in itertools.product(GRID["q"], GRID["p0"], GRID["carry"]):
        f = _pregame_frame(*_filter(sub, q, p0, carry, OBS_SD0))
        f = f[f.season.isin(tune_seasons)]
        home = (~f.game_id.map(tune_g.set_index("game_id").is_neutral.astype(bool))).to_numpy(float)
        err = f.game_id.map(y).to_numpy(float) - (f.S_home - f.S_away).to_numpy(float) - hca_tune * home
        mse = float(np.mean(err**2))
        if best is None or mse < best[0]:
            best = (mse, q, p0, carry)
    assert best is not None
    mse, q, p0, carry = best
    f = _pregame_frame(*_filter(sub, q, p0, carry, OBS_SD0))
    f = f[f.season.isin(tune_seasons)]
    home = (~f.game_id.map(tune_g.set_index("game_id").is_neutral.astype(bool))).to_numpy(float)
    innov = f.game_id.map(y).to_numpy(float) - (f.S_home - f.S_away).to_numpy(float) - f.hca.to_numpy(float) * home
    pred_var = (f.var_home + f.var_away - 2 * f.cov_ha).to_numpy(float) + OBS_SD0**2
    c2 = float(np.mean(innov**2) / np.mean(pred_var))
    return {
        "q": q * c2,
        "p0": p0 * c2,
        "carry": carry,
        "obs_sd": OBS_SD0 * float(np.sqrt(c2)),
        "tune_mse": mse,
        "grid_choice": {"q": q, "p0": p0, "carry": carry, "obs_sd": OBS_SD0},
        "variance_scale": c2,
    }
