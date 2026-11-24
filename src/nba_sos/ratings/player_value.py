"""Player value (PV): official NBA box-score/advanced stats, era-standardised within each
season, combined with weights calibrated on team strength.

Inputs per player-season (regular season, stats.nba.com via nba_api):
  per 100 possessions: PTS, FG3A, FTA, AST, TOV, OREB, DREB, STL, BLK, PF
  advanced: TS% (as (TS - league TS) * USG, i.e. efficiency on volume), USG%, PIE
  impact: box-score on/off per 48 (``players.player_values``) and on-court NET_RATING
          minus the team's net rating

Era standardisation: each feature is converted to a minutes-weighted z-score within its
season (league scoring rose ~9 points per 100 possessions and three-point volume from 26%
to 42% of attempts since 2013-14, so raw rate stats are not comparable across years).

Calibration (BPM-style, but fit on our own team strengths): for team-season t

    theta_sched[t] = c + sum_p m_p * (beta . z_p) + e,    m_p = MIN_p / (team MIN / 5)

where theta_sched is the schedule-adjusted team strength with no health terms (actual
minutes already reflect who played). Ridge penalty chosen by 5-fold CV over team-seasons.
PV rate (per 100 possessions, team-strength units) = beta . z_p, shrunk toward 0 by
MIN / (MIN + SHRINK_MIN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingestion import nba_stats
from ..ingestion.raw_store import RawStore

PER100 = ["PTS", "FG3A", "FTA", "AST", "TOV", "OREB", "DREB", "STL", "BLK", "PF"]
FEATURES = [f"{c.lower()}_100" for c in PER100] + ["ts_x_usg", "usg", "pie", "on_off48", "on_court_rel"]
# Box-score-only features for the lineup-calibrated prior (on/off excluded: it shares
# credit with teammates; impact beyond the box score is left to the lineup RAPM step).
BOX_FEATURES = [f for f in FEATURES if f not in ("on_off48", "on_court_rel")]
MIN_FOR_NORMS = 250.0
SHRINK_MIN = 500.0
ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0)


def season_player_stats(store: RawStore, season: str, prefix: str = "") -> pd.DataFrame:
    """``prefix``: "" for the regular season, "playoff_" for the same stats over the playoffs."""
    a = nba_stats.load(store, f"{prefix}player_advanced", season)
    b = nba_stats.load(store, f"{prefix}player_per100", season)
    d = a[["PLAYER_ID", "PLAYER_NAME", "MIN", "GP", "TS_PCT", "USG_PCT", "PIE", "NET_RATING"]].merge(
        b[["PLAYER_ID"] + PER100], on="PLAYER_ID")
    d = d.rename(columns={c: f"{c.lower()}_100" for c in PER100})
    d = d.rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name", "MIN": "min_total",
                          "GP": "gp", "TS_PCT": "ts", "USG_PCT": "usg", "PIE": "pie", "NET_RATING": "on_court_net"})
    # the Advanced measure reports MIN per game even with per_mode "Totals"
    d["min_total"] = d.min_total * d.gp
    d["season"] = season
    return d


def _wz(x: pd.Series, w: pd.Series, ref: pd.Series) -> pd.Series:
    m = np.average(x[ref], weights=w[ref])
    sd = np.sqrt(np.average((x[ref] - m) ** 2, weights=w[ref]))
    return (x - m) / sd if sd > 0 else x * 0.0


def standardise(stats: pd.DataFrame, onoff: pd.DataFrame, team_net: pd.DataFrame) -> pd.DataFrame:
    """Per-season minutes-weighted z-scores of every feature."""
    d = stats.merge(onoff, on=["season", "player_id"], how="left")
    d["on_off48"] = d.on_off48.fillna(0.0)
    d = d.merge(team_net, on=["season", "player_id"], how="left")
    d["on_court_rel"] = (d.on_court_net - d.team_net).fillna(0.0)
    out = []
    for _s, g in d.groupby("season"):
        g = g.copy()
        ref = g.min_total >= MIN_FOR_NORMS
        lg_ts = np.average(g.ts[ref], weights=g.min_total[ref])
        g["ts_x_usg"] = (g.ts - lg_ts) * g.usg
        for f in FEATURES:
            g[f"z_{f}"] = _wz(g[f].astype(float), g.min_total.astype(float), ref)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def team_design(z: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Per team-season: sum_p m_p * z_p for every feature (m_p sums to 5)."""
    mins = player_games.groupby(["season", "team_id", "player_id"]).minutes.sum().rename("min_team").reset_index()
    tot = mins.groupby(["season", "team_id"]).min_team.sum().rename("team_min")
    mins = mins.join(tot, on=["season", "team_id"])
    mins["m"] = mins.min_team / (mins.team_min / 5)
    x = mins.merge(z[["season", "player_id"] + [f"z_{f}" for f in FEATURES]], on=["season", "player_id"], how="left")
    zc = [f"z_{f}" for f in FEATURES]
    x[zc] = x[zc].fillna(0.0).multiply(x.m, axis=0)
    return x.groupby(["season", "team_id"])[zc].sum().reset_index()


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    pen = np.diag([0.0] + [alpha] * X.shape[1])
    return np.linalg.solve(Xc.T @ Xc + pen, Xc.T @ y)


def fit_weights(design: pd.DataFrame, theta: pd.DataFrame, seed: int = 0) -> dict:
    """Ridge on team-seasons; alpha by 5-fold cross-validation over team-seasons."""
    d = design.merge(theta[["season", "team_id", "theta"]], on=["season", "team_id"])
    zc = [f"z_{f}" for f in FEATURES]
    X, y = d[zc].to_numpy(float), d.theta.to_numpy(float)
    fold = np.random.default_rng(seed).permutation(len(y)) % 5
    cv = {}
    for a in ALPHAS:
        err = []
        for k in range(5):
            tr, te = fold != k, fold == k
            b = _ridge(X[tr], y[tr], a)
            err.append(((y[te] - (b[0] + X[te] @ b[1:])) ** 2).mean())
        cv[a] = float(np.mean(err))
    alpha = min(cv, key=cv.get)
    b = _ridge(X, y, alpha)
    pred = b[0] + X @ b[1:]
    return {"alpha": alpha, "cv_mse": cv, "intercept": float(b[0]), "beta": dict(zip(FEATURES, b[1:])),
            "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "cv_r2": float(1 - cv[alpha] / y.var()), "n_team_seasons": int(len(y))}


def player_pv(z: pd.DataFrame, weights: dict) -> pd.DataFrame:
    beta = np.array([weights["beta"][f] for f in FEATURES])
    out = z[["season", "player_id", "player_name", "min_total", "gp"]].copy()
    raw = z[[f"z_{f}" for f in FEATURES]].to_numpy(float) @ beta
    out["pv_raw"] = raw
    out["pv_rate"] = raw * z.min_total / (z.min_total + SHRINK_MIN)
    return out


def onoff_by_season(values: pd.DataFrame) -> pd.DataFrame:
    """Minutes-weighted on/off per 48 across a player's stints in a season."""
    v = values[values.min_on > 0]
    g = v.assign(w=v.on_off48 * v.min_on).groupby(["season", "player_id"])
    return (g.w.sum() / g.min_on.sum()).rename("on_off48").reset_index()


def team_net_for_players(games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Each player's team net rating (minutes-weighted over his teams) for on_court_rel."""
    t = pd.concat([games[["season", "home_team_id", "home_score", "away_score", "possessions"]].set_axis(
        ["season", "team_id", "pf", "pa", "poss"], axis=1),
        games[["season", "away_team_id", "away_score", "home_score", "possessions"]].set_axis(
        ["season", "team_id", "pf", "pa", "poss"], axis=1)])
    t[["pf", "pa", "poss"]] = t[["pf", "pa", "poss"]].astype(float)
    tn = t.groupby(["season", "team_id"]).apply(lambda x: (x.pf.sum() - x.pa.sum()) / x.poss.sum() * 100,
                                                include_groups=False).rename("tn").reset_index()
    tn["team_id"] = tn.team_id.astype(int)
    m = player_games.groupby(["season", "player_id", "team_id"]).minutes.sum().rename("mins").reset_index()
    m["team_id"] = m.team_id.astype(int)
    m = m.merge(tn, on=["season", "team_id"])
    g = m.assign(w=m.tn * m.mins).groupby(["season", "player_id"])
    return (g.w.sum() / g.mins.sum()).rename("team_net").reset_index()


# ---------------------------------------------------------------------------------------
# Absence-calibrated value: weights identified from games a player missed, within the same
# team-season (team-season fixed effects absorb team quality, so box-score "accounting"
# cannot stand in for individual impact).
#
#   y_g = theta[h,s] - theta[a,s] + HCA_s + b_b2b*d_b2b + b_min*d(sum_missing f)
#         + sum_k gamma_k * d(sum_missing f * z_k) + e
#
# A player's value on the floor relative to an average player (per 100 possessions):
#   pv_abs = -(gamma . z_p)     (b_min is the common replacement-level step)
# ---------------------------------------------------------------------------------------
def absence_features(rg: pd.DataFrame, z: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: sum over missing roster players of f and f*z_k."""
    m = rg[~rg.played].merge(values[["season", "player_id", "team_id", "mpg"]], on=["season", "player_id", "team_id"])
    m = m.merge(z[["season", "player_id"] + [f"z_{f}" for f in FEATURES]], on=["season", "player_id"], how="left")
    zc = [f"z_{f}" for f in FEATURES]
    m[zc] = m[zc].fillna(0.0)
    m["f"] = m.mpg / 48
    fz = m[zc].multiply(m.f, axis=0)
    fz["f"] = m.f
    fz[["game_id", "team_id"]] = m[["game_id", "team_id"]]
    return fz.groupby(["game_id", "team_id"]).sum().reset_index()


def fit_absence(gf: pd.DataFrame, absf: pd.DataFrame, alpha_team: float = 10.0, alpha_gamma: float = 10.0) -> dict:
    zc = [f"z_{f}" for f in FEATURES]
    d = gf.copy()
    a = absf.copy()
    a["team_id"] = a.team_id.astype(int)
    for side in ("home", "away"):
        x = a.rename(columns={"team_id": f"{side}_team_id", **{c: f"{side}_{c}" for c in zc + ["f"]}})
        d = d.assign(**{f"{side}_team_id": d[f"{side}_team_id"].astype(int)}).merge(
            x, on=["game_id", f"{side}_team_id"], how="left")
    for c in zc + ["f"]:
        d[f"d_{c}"] = d[f"home_{c}"].fillna(0.0) - d[f"away_{c}"].fillna(0.0)
    keys = sorted(set(d.home_key) | set(d.away_key))
    idx = {k: i for i, k in enumerate(keys)}
    seasons = sorted(d.season.unique())
    n, k = len(d), len(keys)
    extra = ["d_b2b", "d_f"] + [f"d_{c}" for c in zc]
    X = np.zeros((n, k + len(seasons) + len(extra)))
    X[np.arange(n), d.home_key.map(idx).to_numpy()] = 1.0
    X[np.arange(n), d.away_key.map(idx).to_numpy()] = -1.0
    X[np.arange(n), k + d.season.map({s: i for i, s in enumerate(seasons)}).to_numpy()] = d.home_flag.to_numpy(float)
    X[:, k + len(seasons):] = d[extra].to_numpy(float)
    pen = np.zeros(X.shape[1])
    pen[:k] = alpha_team
    pen[k + len(seasons) + 2:] = alpha_gamma
    y = d.y.to_numpy(float)
    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    off = k + len(seasons)
    gamma = dict(zip(FEATURES, beta[off + 2:]))
    resid = y - X @ beta
    return {"gamma": gamma, "b_b2b": float(beta[off]), "b_min": float(beta[off + 1]),
            "sigma": float(resid.std()), "n_games": int(n)}


def player_pv_abs(z: pd.DataFrame, fit: dict) -> pd.DataFrame:
    g = np.array([fit["gamma"][f] for f in FEATURES])
    out = z[["season", "player_id", "player_name", "min_total", "gp"]].copy()
    out["pv_abs_raw"] = -(z[[f"z_{f}" for f in FEATURES]].to_numpy(float) @ g)
    out["pv_abs"] = out.pv_abs_raw * z.min_total / (z.min_total + SHRINK_MIN)
    return out


# ---------------------------------------------------------------------------------------
# Blended value (final PV). Two inputs per player-season, both era-standardised:
#   pv_box  — box-score production value (team-level calibration above; PIE-dominated)
#   z_onoff — box-score on/off z-score (impact; noisy, shares credit with teammates)
# Their mix is estimated only from games a player missed (team-season fixed effects):
#   y_g = ... + b_min*d(sum f) + g_box*d(sum f*pv_box) + g_onoff*d(sum f*z_onoff)
# pv = -(g_box*pv_box + g_onoff*z_onoff), shrunk by minutes. Two weights -> little room to overfit.
# ---------------------------------------------------------------------------------------
BLEND_INPUTS = ["pv_box", "z_onoff"]


def blend_inputs(z: pd.DataFrame, weights_box: dict) -> pd.DataFrame:
    beta = np.array([weights_box["beta"][f] for f in FEATURES])
    out = z[["season", "player_id", "player_name", "min_total", "gp"]].copy()
    out["pv_box"] = z[[f"z_{f}" for f in FEATURES]].to_numpy(float) @ beta
    out["z_onoff"] = z["z_on_off48"]
    return out


def absence_features_blend(rg: pd.DataFrame, inputs: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    m = rg[~rg.played].merge(values[["season", "player_id", "team_id", "mpg"]], on=["season", "player_id", "team_id"])
    m = m.merge(inputs[["season", "player_id"] + BLEND_INPUTS], on=["season", "player_id"], how="left")
    m[BLEND_INPUTS] = m[BLEND_INPUTS].fillna(0.0)
    m["f"] = m.mpg / 48
    out = m[BLEND_INPUTS].multiply(m.f, axis=0)
    out["f"] = m.f
    out[["game_id", "team_id"]] = m[["game_id", "team_id"]]
    return out.groupby(["game_id", "team_id"]).sum().reset_index()


def fit_blend(gf: pd.DataFrame, absf: pd.DataFrame, alpha_team: float = 10.0) -> dict:
    d = gf.copy()
    a = absf.assign(team_id=absf.team_id.astype(int))
    cols = BLEND_INPUTS + ["f"]
    for side in ("home", "away"):
        x = a.rename(columns={"team_id": f"{side}_team_id", **{c: f"{side}_{c}" for c in cols}})
        d = d.assign(**{f"{side}_team_id": d[f"{side}_team_id"].astype(int)}).merge(
            x, on=["game_id", f"{side}_team_id"], how="left")
    for c in cols:
        d[f"d_{c}"] = d[f"home_{c}"].fillna(0.0) - d[f"away_{c}"].fillna(0.0)
    keys = sorted(set(d.home_key) | set(d.away_key))
    idx = {k: i for i, k in enumerate(keys)}
    seasons = sorted(d.season.unique())
    n, k = len(d), len(keys)
    extra = ["d_b2b", "d_f"] + [f"d_{c}" for c in BLEND_INPUTS]
    X = np.zeros((n, k + len(seasons) + len(extra)))
    X[np.arange(n), d.home_key.map(idx).to_numpy()] = 1.0
    X[np.arange(n), d.away_key.map(idx).to_numpy()] = -1.0
    X[np.arange(n), k + d.season.map({s: i for i, s in enumerate(seasons)}).to_numpy()] = d.home_flag.to_numpy(float)
    X[:, k + len(seasons):] = d[extra].to_numpy(float)
    pen = np.zeros(X.shape[1])
    pen[:k] = alpha_team
    y = d.y.to_numpy(float)
    A = X.T @ X + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ y)
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(A)) * sigma2)
    off = k + len(seasons)
    return {"g_box": float(beta[off + 2]), "g_onoff": float(beta[off + 3]),
            "se_box": float(se[off + 2]), "se_onoff": float(se[off + 3]),
            "b_min": float(beta[off + 1]), "b_b2b": float(beta[off]), "n_games": int(n)}


def player_pv_blend(inputs: pd.DataFrame, fit: dict) -> pd.DataFrame:
    out = inputs[["season", "player_id", "player_name", "min_total", "gp", "pv_box", "z_onoff"]].copy()
    raw = -(fit["g_box"] * inputs.pv_box + fit["g_onoff"] * inputs.z_onoff)
    out["pv_raw"] = raw
    out["pv"] = raw * inputs.min_total / (inputs.min_total + SHRINK_MIN)
    return out
