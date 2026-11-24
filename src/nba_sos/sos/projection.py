"""Remaining-schedule difficulty (plan section 9) and joint league Monte Carlo
(plan section 10), from the information available at a cutoff date.

Game model (home margin per 100 possessions), fit on earlier seasons only:

    y = b0 * phi(h) * (S_home - S_away) + sum_j c_j dI_j
        + hca * venue + sum_k c_k (x_k,home - x_k,away) + e

  S        dynamic opponent-adjusted strength as of the cutoff (roster-based ridge prior)
  h        days from the cutoff to the game; phi(h) = exp(-r h) (regression toward the mean)
  I        availability components (recently missing, expected missing of players out at
           the cutoff, expected new injuries; :mod:`nba_sos.sos.availability`) at each
           team's k-th next game; which ones enter is chosen by season-grouped k-fold
  venue    1 at the home team's court, 0 on a neutral floor
  x        schedule load of each side, known from the schedule. Candidates (``LOAD``) are
           chosen by forward selection on season-grouped k-fold out-of-fold error inside the
           training seasons; the home-court coefficient is estimated per season and the
           forecast uses the mean of the last ``HCA_SEASONS``.
  e        N(0, sigma_game^2)

Uncertainty about where each team really is, growing with the horizon: team j's latent
strength at horizon h is b0 phi(h) S_j + u_j + w_j(h), u_j ~ N(0, kappa sd_j^2) (sd_j: the
ridge posterior sd) and w_j a random walk with variance q per day; kappa and q come from
the covariance of the same team's residuals after weekly cutoffs of earlier seasons.

Schedule difficulty of a set of team-games (plan 9.1-9.3): an average team (strength 0,
known exactly, healthy) takes the team's place, keeps its venue and its own schedule load,
and faces the real opponent (strength at h with its uncertainty and expected availability,
the opponent's schedule load). EW_ref = sum P(win); D-SOS = G/2 - EW_ref. Opponent SOS keeps
only the opponent's strength. Exact Shapley parts over
  opponent | opp_availability | venue | own_rest | own_travel | opp_fatigue
add up to D-SOS (load terms centred on the league-average load).

Monte Carlo: all remaining games in date order, both teams of a game sharing one result;
each path draws u, the random walk, and every injured player's return game. NBA Cup
knockout games and the extra games are simulated from the simulated group tables
(:mod:`nba_sos.sos.cup`); seeds use the official tie-break procedures
(:mod:`nba_sos.sos.standings`); 7-10 play in (7v8, loser v winner of 9v10).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from ..evaluation.metrics import season_folds
from . import availability as AV
from . import cup as CUP
from .standings import Season

LOAD = {  # name: category
    "b2b": "rest", "rest3": "rest", "g4": "rest", "g7": "rest", "short_turn": "rest",
    "km": "travel", "tz_abs": "travel", "tz_east": "travel", "road": "travel", "road_b2b": "travel",
    "body_early": "travel", "body_late": "travel",
}
HCA_SEASONS = 3
COMPONENTS = ["opponent", "opp_availability", "venue", "own_rest", "own_travel", "opp_fatigue"]
INJ = ["rec_v", "rec_f", "ret_v", "ret_f", "new_v", "new_f"]
# availability specifications compared by season-grouped k-fold (projection.horizon_cv)
INJ_SPECS = {"none": [], "rec+ret (value)": ["rec_v", "ret_v"],
             "rec+ret (value, minutes)": ["rec_v", "ret_v", "rec_f", "ret_f"], "all": INJ}


# ------------------------------------------------------------------ inputs
def schedule_load(f: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team_id) candidate schedule-load features (all known from the schedule).
      b2b         second night of a back-to-back
      rest3       days of rest, capped at 3
      g4, g7      games in the last 4 / 7 days (incl. this one)
      short_turn  hours under 26 between tip-offs (late game then early game), / 10
      km          travel from the previous game's venue, / 1000
      tz_abs      |time-zone change|;  tz_east: eastward change (hours, >= 0)
      road        consecutive road games, capped at 6
      road_b2b    back-to-back with both legs away
      body_early  hours the tip-off is before 19:00 on the team's home clock
      body_late   hours after 22:00 on the team's home clock"""
    b2b = f.is_back_to_back.astype("boolean").fillna(False).astype(float)
    tz = f.timezone_change_hours.astype(float).fillna(0.0)
    body = f.body_clock_tipoff_hour.astype(float)
    turn = f.turnaround_hours.astype(float)
    return pd.DataFrame({
        "game_id": f.game_id.to_numpy(), "team_id": f.team_id.astype(int).to_numpy(),
        "b2b": b2b.to_numpy(), "rest3": f.rest_days.astype(float).clip(upper=3).fillna(3.0).to_numpy(),
        "g4": f.games_last_4_days.astype(float).fillna(1.0).to_numpy(),
        "g7": f.games_last_7_days.astype(float).fillna(1.0).to_numpy(),
        "short_turn": ((26 - turn).clip(lower=0) / 10).fillna(0.0).to_numpy(),
        "km": (f.travel_distance_km.astype(float).fillna(0.0) / 1000).to_numpy(),
        "tz_abs": tz.abs().to_numpy(), "tz_east": tz.clip(lower=0).to_numpy(),
        "road": f.consecutive_road_games.astype(float).clip(upper=6).fillna(0.0).to_numpy(),
        "road_b2b": (f.b2b_type.astype(object) == "AA").astype(float).to_numpy(),
        "body_early": (19 - body).clip(lower=0).fillna(0.0).to_numpy(),
        "body_late": (body - 22).clip(lower=0).fillna(0.0).to_numpy()})


def game_rows(games: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    """games (game_id, game_date, home/away ids, is_neutral[, Cup labels]) + load of both sides."""
    keep = ["game_id", "season", "game_date", "home_team_id", "away_team_id", "is_neutral"]
    keep += [c for c in ("game_sublabel", "game_subtype") if c in games.columns]
    g = games[keep].copy()
    g["home_team_id"], g["away_team_id"] = g.home_team_id.astype(int), g.away_team_id.astype(int)
    g["date"] = pd.to_datetime(g.game_date)
    for side in ("home", "away"):
        side_load = load.rename(columns={"team_id": f"{side}_team_id", **{k: f"{k}_{side}" for k in LOAD}})
        g = g.merge(side_load, on=["game_id", f"{side}_team_id"], how="left")
    for k in LOAD:
        for side in ("home", "away"):
            g[f"{k}_{side}"] = g[f"{k}_{side}"].fillna(0.0)
    g["venue"] = (~g.is_neutral.astype(bool)).astype(float)
    return g


# ------------------------------------------------------------------ pre-game model and load selection
def _design(x: pd.DataFrame, feats: list[str], season_hca: bool) -> tuple[np.ndarray, list[str]]:
    cols = [x.dyn_diff.to_numpy(float)]
    names = ["dyn_diff"]
    if season_hca:
        for s in sorted(x.season.unique()):
            cols.append(x.venue.to_numpy(float) * (x.season == s).to_numpy(float))
            names.append(f"hca:{s}")
    else:
        cols.append(x.venue.to_numpy(float))
        names.append("venue")
    for k in feats:
        cols.append((x[f"{k}_home"] - x[f"{k}_away"]).to_numpy(float))
        names.append(k)
    return np.column_stack(cols), names


def fit_pregame(x: pd.DataFrame, feats: list[str]) -> dict:
    """OLS of home margin per 100 on pre-game strength difference, per-season home court and
    the chosen load differences. Forecast home court = mean of the last HCA_SEASONS."""
    d = x.dropna(subset=["dyn_diff", "y"])
    X, names = _design(d, feats, season_hca=True)
    y = d.y.to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    sigma = float(np.sqrt(res @ res / (len(y) - len(b))))
    se = np.sqrt(np.diag(sigma ** 2 * np.linalg.pinv(X.T @ X)))
    coef = dict(zip(names, b))
    hcas = {n.split(":")[1]: float(v) for n, v in coef.items() if n.startswith("hca:")}
    recent = sorted(hcas)[-HCA_SEASONS:]
    return {"b_strength_h0": float(coef["dyn_diff"]), "hca": float(np.mean([hcas[s] for s in recent])),
            "hca_by_season": hcas, "c": {k: float(coef[k]) for k in feats},
            "se": {k: float(s) for k, s in zip(names, se) if not k.startswith("hca:")},
            "sigma": sigma, "n_games": int(len(y))}


def select_load(x: pd.DataFrame, k: int = 5) -> dict:
    """Forward selection of load features on season-grouped k-fold out-of-fold squared error.
    A feature enters when its mean OOF improvement exceeds one standard error of the paired
    per-game difference; stops when none does."""
    d = x.dropna(subset=["dyn_diff", "y"]).reset_index(drop=True)
    folds = season_folds(sorted(d.season.unique()), k)
    y = d.y.to_numpy(float)

    def oof(feats):
        pred = np.empty(len(d))
        for tr, va in folds:
            itr, iva = d.season.isin(tr).to_numpy(), d.season.isin(va).to_numpy()
            Xtr, _ = _design(d[itr], feats, season_hca=False)
            b, *_ = np.linalg.lstsq(Xtr, y[itr], rcond=None)
            Xva, _ = _design(d[iva], feats, season_hca=False)
            pred[iva] = Xva @ b
        return (y - pred) ** 2

    chosen, cur = [], oof([])
    path = [{"step": 0, "added": None, "oof_mse": float(cur.mean())}]
    while True:
        best = None
        for f in [c for c in LOAD if c not in chosen]:
            e = oof(chosen + [f])
            diff = cur - e
            gain, se = float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff)))
            if gain > se and (best is None or gain > best[1]):
                best = (f, gain, se, e)
        if best is None:
            break
        chosen.append(best[0])
        cur = best[3]
        path.append({"step": len(chosen), "added": best[0], "oof_mse": float(cur.mean()), "gain": best[1], "se": best[2]})
    return {"features": chosen, "path": path, "folds": k}


def load_effect(g: pd.DataFrame, c: dict, keys=None) -> np.ndarray:
    keys = list(c) if keys is None else keys
    return sum((c[k] * (g[f"{k}_home"].to_numpy(float) - g[f"{k}_away"].to_numpy(float)) for k in keys),
               np.zeros(len(g)))


# ------------------------------------------------------------------ horizon model
def future_index(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """k_home, k_away: each team's game number counted from the cutoff (1 = next game)."""
    fut = rows[rows.date >= cutoff].sort_values(["date", "game_id"])
    if fut.empty:
        return fut.assign(k_home=pd.Series(dtype=int), k_away=pd.Series(dtype=int))
    side = pd.concat([fut[["game_id", "date", "home_team_id"]].rename(columns={"home_team_id": "team_id"}).assign(s="home"),
                      fut[["game_id", "date", "away_team_id"]].rename(columns={"away_team_id": "team_id"}).assign(s="away")])
    side = side.sort_values(["date", "game_id"])
    side["k"] = side.groupby("team_id").cumcount() + 1
    k = side.pivot(index="game_id", columns="s", values="k")
    return fut.assign(k_home=fut.game_id.map(k["home"]).to_numpy(int), k_away=fut.game_id.map(k["away"]).to_numpy(int))


def horizon_rows(games_rows: pd.DataFrame, strengths: pd.DataFrame, y: pd.Series, injury=None) -> pd.DataFrame:
    """Games after each cutoff with the cutoff strengths of both teams.
    ``strengths``: cutoff, team_id, S, sd. ``injury(cutoff) -> {team: (inj_v[k], inj_f[k])}``."""
    out = []
    for cut, st in strengths.groupby("cutoff"):
        fut = future_index(games_rows, cut)
        if fut.empty:
            continue
        s = st.set_index("team_id")
        r = fut[["game_id", "home_team_id", "away_team_id", "venue", "k_home", "k_away"]
                + [f"{k}_{sd}" for k in LOAD for sd in ("home", "away")]].copy()
        r["cutoff"] = cut
        r["h"] = (fut.date - cut).dt.days.to_numpy(float)
        r["S_home"], r["S_away"] = s.S.reindex(fut.home_team_id).to_numpy(), s.S.reindex(fut.away_team_id).to_numpy()
        r["sd_home"], r["sd_away"] = s.sd.reindex(fut.home_team_id).to_numpy(), s.sd.reindex(fut.away_team_id).to_numpy()
        r["sd2"] = r.sd_home ** 2 + r.sd_away ** 2
        r["y"] = fut.game_id.map(y).to_numpy(float)
        add_injury(r, injury(cut) if injury is not None else None)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def add_injury(r: pd.DataFrame, curves: dict | None) -> None:
    """inj_<component>_home/away columns from :func:`availability.expected_change` output
    (zeros without a state)."""
    for side in ("home", "away"):
        ks = np.minimum(r[f"k_{side}"].to_numpy(int), AV.K_MAX)
        teams = r[f"{side}_team_id"].to_numpy(int)
        for c in INJ:
            if curves is None:
                r[f"inj_{c}_{side}"] = 0.0
            elif c.startswith("rec"):
                r[f"inj_{c}_{side}"] = [curves[t_][c] if t_ in curves else 0.0 for t_ in teams]
            else:
                r[f"inj_{c}_{side}"] = [curves[t_][c][k] if t_ in curves else 0.0 for t_, k in zip(teams, ks)]


def fit_horizon(rows: pd.DataFrame, pre: dict, inj_cols: list[str] | None = None) -> dict:
    """1. mean: r by 1-D search; given r, b0 (and the availability coefficients of
       ``inj_cols``) by least squares on the margin net of the fixed home/load terms.
    2. team-level error: with e_jg the residual of team j's game g (team j's view), for
       g != g' of the same team and cutoff E[e_jg e_jg'] = kappa sd_j^2 + q min(h_g, h_g')
       (different opponents' errors are independent), so per (cutoff, team)
           (sum e)^2 - sum e^2 = kappa sd_j^2 G(G-1) + q 2 sum_{g<g'} min(h_g, h_g') + noise
       and kappa, q come from least squares over all (cutoff, team) cells.
    3. game noise: sigma_game^2 = mean(e^2 - kappa sd2 - 2 q h)."""
    dS = (rows.S_home - rows.S_away).to_numpy(float)
    base = pre["hca"] * rows.venue.to_numpy(float) + load_effect(rows, pre["c"])
    h, y = rows.h.to_numpy(float), rows.y.to_numpy(float)
    inj_cols = inj_cols or []
    extra = [(rows[f"inj_{c}_home"] - rows[f"inj_{c}_away"]).to_numpy(float) for c in inj_cols]

    def solve(r):
        X = np.column_stack([np.exp(-r * h) * dS] + extra)
        b, *_ = np.linalg.lstsq(X, y - base, rcond=None)
        return b, X

    def sse(logr):
        b, X = solve(math.exp(logr))
        return float(np.mean((y - base - X @ b) ** 2))

    res = minimize_scalar(sse, bounds=(math.log(1e-6), math.log(0.05)), method="bounded")
    r = float(math.exp(res.x))
    b, X = solve(r)
    e = y - base - X @ b
    side = pd.concat([
        pd.DataFrame({"cutoff": rows.cutoff.to_numpy(), "team_id": rows.home_team_id.to_numpy(), "e": e, "h": h,
                      "sd": rows.sd_home.to_numpy(float)}),
        pd.DataFrame({"cutoff": rows.cutoff.to_numpy(), "team_id": rows.away_team_id.to_numpy(), "e": -e, "h": h,
                      "sd": rows.sd_away.to_numpy(float)})], ignore_index=True)
    side = side.sort_values(["cutoff", "team_id", "h"]).reset_index(drop=True)
    grp = side.groupby(["cutoff", "team_id"], sort=False)
    G = grp.e.size().to_numpy(float)
    s_e = grp.e.sum().to_numpy()
    s_e2 = side.assign(e2=side.e ** 2).groupby(["cutoff", "team_id"], sort=False).e2.sum().to_numpy()
    sd2 = grp.sd.first().to_numpy(float) ** 2
    side["k"] = grp.cumcount()
    side["G"] = grp.e.transform("size")
    side["minsum"] = side.h * (side.G - 1 - side.k)            # sum_{g<g'} min(h_g, h_g') over sorted h
    minsum = side.groupby(["cutoff", "team_id"], sort=False).minsum.sum().to_numpy(float)
    M = np.column_stack([sd2 * G * (G - 1), 2 * minsum])
    coef, *_ = np.linalg.lstsq(M, s_e ** 2 - s_e2, rcond=None)
    kappa, q = (max(float(c), 0.0) for c in coef)
    sigma_game = float(np.sqrt(np.mean(e ** 2 - kappa * rows.sd2.to_numpy(float) - 2 * q * h)))
    out = {"b0": float(b[0]), "r_per_day": r, "kappa": kappa, "q_per_day": q, "kappa_raw": float(coef[0]),
           "q_raw": float(coef[1]), "sigma_game": sigma_game, "n_rows": int(len(rows)), "n_cells": int(len(G)),
           "mse": float(np.mean(e ** 2))}
    if inj_cols:
        out["c_inj"] = {c: float(v) for c, v in zip(inj_cols, b[1:])}
    return out


def mean_margin(rows: pd.DataFrame, pre: dict, hz: dict) -> np.ndarray:
    m = (hz["b0"] * np.exp(-hz["r_per_day"] * rows.h.to_numpy(float)) * (rows.S_home - rows.S_away).to_numpy(float)
         + pre["hca"] * rows.venue.to_numpy(float) + load_effect(rows, pre["c"]))
    for c, v in hz.get("c_inj", {}).items():
        m = m + v * (rows[f"inj_{c}_home"] - rows[f"inj_{c}_away"]).to_numpy(float)
    return m


def horizon_cv(rows: pd.DataFrame, pre: dict, k: int = 5) -> dict:
    """Season-grouped k-fold out-of-fold margin MSE for every availability specification
    (rows carry a ``season`` column). The chosen spec is the best one whose mean OOF
    improvement over "none" exceeds one standard error of the paired per-game difference."""
    folds = season_folds(sorted(rows.season.unique()), k)
    y = rows.y.to_numpy(float)
    err = {}
    for name, cols in INJ_SPECS.items():
        e = np.empty(len(rows))
        for tr, va in folds:
            itr, iva = rows.season.isin(tr).to_numpy(), rows.season.isin(va).to_numpy()
            hz = fit_horizon(rows[itr], pre, cols)
            e[iva] = (y[iva] - mean_margin(rows[iva], pre, hz)) ** 2
        err[name] = e
    out = {"oof_mse": {n: float(e.mean()) for n, e in err.items()}, "chosen": "none"}
    best = 0.0
    for n, e in err.items():
        d = err["none"] - e
        gain, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
        out.setdefault("gain_vs_none", {})[n] = {"gain": gain, "se": se}
        if n != "none" and gain > se and gain > best:
            best, out["chosen"] = gain, n
    return out


# ------------------------------------------------------------------ schedule difficulty
def team_games(g: pd.DataFrame) -> pd.DataFrame:
    """Game rows -> one row per (team, game) from the team's perspective."""
    inj = [f"inj_{c}" for c in INJ if f"inj_{c}_home" in g]
    h = pd.DataFrame({"game_id": g.game_id, "date": g.date, "team_id": g.home_team_id, "opponent_id": g.away_team_id,
                      "venue": g.venue, **{f"{k}_own": g[f"{k}_home"] for k in LOAD},
                      **{f"{k}_opp": g[f"{k}_away"] for k in LOAD}, **{f"{c}_opp": g[f"{c}_away"] for c in inj}})
    a = pd.DataFrame({"game_id": g.game_id, "date": g.date, "team_id": g.away_team_id, "opponent_id": g.home_team_id,
                      "venue": -g.venue, **{f"{k}_own": g[f"{k}_away"] for k in LOAD},
                      **{f"{k}_opp": g[f"{k}_home"] for k in LOAD}, **{f"{c}_opp": g[f"{c}_home"] for c in inj}})
    return pd.concat([h, a], ignore_index=True)


def _ref_mu_parts(tg: pd.DataFrame, s: pd.DataFrame, pre: dict, hz: dict, cutoff: pd.Timestamp) -> dict:
    """Margin parts of the average team. Schedule load enters relative to the league-average
    load of these team-games, so a team with a typical schedule gets ~0 for its own load and
    for its opponents' (the sum, and hence D-SOS, does not depend on this centring)."""
    h = (tg.date - cutoff).dt.days.to_numpy(float).clip(min=0)
    c = pre["c"]
    opp = s.reindex(tg.opponent_id)
    xbar = {k: float(tg[f"{k}_own"].mean()) for k in c}

    def load(side, cat):
        return sum((c[k] * (tg[f"{k}_{side}"].to_numpy(float) - xbar[k]) for k in c if cat is None or LOAD[k] == cat),
                   np.zeros(len(tg)))

    # availability terms only move teams relative to each other in a game; the reference
    # team is league-average in them too, so they are centred on the league mean of the
    # same week (new injuries and returns drift with the horizon for every team)
    avail = np.zeros(len(tg))
    week = tg.date.dt.to_period("W").astype(str)
    for comp, v in hz.get("c_inj", {}).items():
        x = tg[f"inj_{comp}_opp"].astype(float)
        avail = avail - v * (x - x.groupby(week).transform("mean")).to_numpy(float)
    return {
        "opponent": -hz["b0"] * np.exp(-hz["r_per_day"] * h) * opp.S.to_numpy(float),
        "opp_availability": avail,
        "venue": pre["hca"] * tg.venue.to_numpy(float),
        "own_rest": load("own", "rest"),
        "own_travel": load("own", "travel"),
        "opp_fatigue": -load("opp", None),
        "var": hz["sigma_game"] ** 2 + hz["kappa"] * opp.sd.to_numpy(float) ** 2 + hz["q_per_day"] * h,
    }


def schedule_difficulty(tg: pd.DataFrame, strengths: pd.DataFrame, pre: dict, hz: dict,
                        cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per team: games, EW_ref, D-SOS, opponent SOS, Shapley parts of D-SOS."""
    s = strengths.set_index("team_id")
    parts = _ref_mu_parts(tg, s, pre, hz, cutoff)
    sd = np.sqrt(parts["var"])
    teams = np.unique(tg.team_id.to_numpy())
    idx = np.searchsorted(teams, tg.team_id.to_numpy())
    n = np.bincount(idx, minlength=len(teams)).astype(float)

    def dsos(subset) -> np.ndarray:
        mu = sum((parts[k] for k in subset), np.zeros(len(tg)))
        return n / 2 - np.bincount(idx, weights=norm.cdf(mu / sd), minlength=len(teams))

    value = {frozenset(S): dsos(S) for r in range(len(COMPONENTS) + 1) for S in itertools.combinations(COMPONENTS, r)}
    m = len(COMPONENTS)
    shap = {}
    for k in COMPONENTS:
        others = [c for c in COMPONENTS if c != k]
        tot = np.zeros(len(teams))
        for r in range(m):
            wgt = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
            for S in itertools.combinations(others, r):
                tot += wgt * (value[frozenset(S) | {k}] - value[frozenset(S)])
        shap[k] = tot
    full = value[frozenset(COMPONENTS)]
    out = pd.DataFrame({"team_id": teams, "games": n.astype(int), "ew_ref": n / 2 - full, "d_sos": full,
                        "opp_sos": value[frozenset({"opponent"})], **{f"part_{k}": v for k, v in shap.items()}})
    out["sos_rate"] = 1 - out.ew_ref / out.games.replace(0, np.nan)
    out["d_sos_rank"] = out.d_sos.rank(ascending=False, method="min").astype(int)
    return out


# ------------------------------------------------------------------ simulation
class _League:
    """Per-path league state: wins, head-to-head wins W[p, i, j] (i beat j), point differential."""

    def __init__(self, n_paths, teams, done: pd.DataFrame | None):
        self.T = len(teams)
        self.pos = {t: i for i, t in enumerate(teams)}
        self.wins = np.zeros((n_paths, self.T))
        self.W = np.zeros((n_paths, self.T, self.T), dtype=np.int16)
        self.pdiff = np.zeros((n_paths, self.T))
        if done is not None and len(done):
            hi, ai = done.home_team_id.map(self.pos).to_numpy(), done.away_team_id.map(self.pos).to_numpy()
            m = (done.home_score.astype(float) - done.away_score.astype(float)).to_numpy()
            for h, a, mm in zip(hi, ai, m):
                w, l_ = (h, a) if mm > 0 else (a, h)
                self.wins[:, w] += 1
                self.W[:, w, l_] += 1
                self.pdiff[:, h] += mm
                self.pdiff[:, a] -= mm

    def record(self, hi, ai, margin_pts):
        """hi, ai broadcastable to margin_pts (paths, games); margin in points, home view."""
        P = self.wins.shape[0]
        hi = np.broadcast_to(hi, margin_pts.shape)
        ai = np.broadcast_to(ai, margin_pts.shape)
        hw = margin_pts > 0
        rows = np.broadcast_to(np.arange(P)[:, None], margin_pts.shape)
        w, l_ = np.where(hw, hi, ai), np.where(hw, ai, hi)
        np.add.at(self.wins, (rows, w), 1)
        np.add.at(self.W, (rows, w, l_), 1)
        np.add.at(self.pdiff, (rows, hi), margin_pts)
        np.add.at(self.pdiff, (rows, ai), -margin_pts)
        return hw

    def record_one(self, p: int, h: int, a: int, m: float) -> None:
        w, l_ = (h, a) if m > 0 else (a, h)
        self.wins[p, w] += 1
        self.W[p, w, l_] += 1
        self.pdiff[p, h] += m
        self.pdiff[p, a] -= m


def simulate(rem: pd.DataFrame, strengths: pd.DataFrame, pre: dict, hz: dict, cutoff: pd.Timestamp,
             done: pd.DataFrame | None = None, n_paths: int = 10000, seed: int = 0,
             conference: pd.Series | None = None, division: pd.Series | None = None,
             injuries: dict | None = None, cup: CUP.CupPlan | None = None, cup_done: pd.DataFrame | None = None,
             poss: float = 100.0, tg_ref: pd.DataFrame | None = None) -> dict:
    """Joint simulation of the remaining games ``rem`` (game rows).
    ``done``: completed games (home/away ids and scores) for records and tie-breaks.
    ``injuries``: :func:`availability.sample_paths` output. ``cup``: simulate the NBA Cup
    knockout stage and extra games (their rows must not be in ``rem``); ``cup_done``:
    completed group games with scores."""
    rng = np.random.default_rng(seed)
    s = strengths.set_index("team_id")
    teams = np.array(sorted(s.index))
    lg = _League(n_paths, teams, done)
    T, pos = lg.T, lg.pos
    S0, sd0 = s.S.reindex(teams).to_numpy(float), s.sd.reindex(teams).to_numpy(float)
    sigma, hca, to_pts = hz["sigma_game"], pre["hca"], poss / 100.0
    u = rng.normal(0.0, 1.0, (n_paths, T)) * np.sqrt(hz["kappa"]) * sd0
    rem = rem.sort_values(["date", "game_id"])
    hi, ai = rem.home_team_id.map(pos).to_numpy(), rem.away_team_id.map(pos).to_numpy()
    fixed = hca * rem.venue.to_numpy(float) + load_effect(rem, pre["c"])
    days = ((rem.date - cutoff).dt.days.to_numpy(float)).clip(min=0)
    rem_ids = rem.game_id.to_numpy()
    tdays = {i: np.sort(np.concatenate([days[hi == i], days[ai == i]])) for i in range(T)}
    walk = np.zeros((n_paths, T))
    last_day = 0.0
    rows_p = np.arange(n_paths)

    c_inj = hz.get("c_inj", {})
    inj_on = injuries is not None and bool(c_inj)
    inj = [injuries.get(int(t)) for t in teams] if inj_on else None

    def inj_adj(day) -> np.ndarray:
        """Availability terms per path: rec and new_* at their expectation, ret_* from each
        path's sampled return games."""
        out = np.zeros((n_paths, T))
        if not inj_on:
            return out
        for i in range(T):
            st = inj[i]
            if st is None:
                continue
            k = min(int(np.searchsorted(tdays[i], day, side="left")) + 1, AV.K_MAX)
            still = st["R"] >= k
            for c, v in c_inj.items():
                if c.startswith("rec"):
                    out[:, i] += v * st[c]
                elif c.startswith("new"):
                    out[:, i] += v * st[c][k]
                else:
                    w = st["fv"] if c == "ret_v" else st["f"]
                    out[:, i] += v * (still * w).sum(axis=1)
        # relative terms: centred across the league per path (game margins use differences only)
        return out - out.mean(axis=1, keepdims=True)

    def latent(day):
        return hz["b0"] * math.exp(-hz["r_per_day"] * day) * S0 + u + walk + inj_adj(day)

    def play(L, h, a, home_edge):
        m = (L[rows_p, h] - L[rows_p, a] + hca * home_edge + rng.normal(0.0, sigma, n_paths)) * to_pts
        hw = lg.record(h[:, None], a[:, None], m[:, None])[:, 0]
        return np.where(hw, h, a), np.where(hw, a, h)

    cs = None
    if cup is not None:
        gid = cup.group_games.game_id.tolist()
        cs = {"idx": {g: i for i, g in enumerate(gid)}, "hw": np.zeros((n_paths, len(gid)), dtype=bool),
              "m": np.zeros((n_paths, len(gid))), "ko": {},
              "out": {k: np.zeros((n_paths, T), bool) for k in ("qf", "sf", "final")},
              "days": {k: max(float((d - cutoff).days), 0.0) for k, d in
                       (("qf", cup.qf_date), ("sf", cup.sf_date), ("x0", cup.extra_dates[0]), ("x1", cup.extra_dates[1]))}}
        if cup_done is not None:
            for g_, hs, as_ in zip(cup_done.game_id, cup_done.home_score.astype(float), cup_done.away_score.astype(float)):
                if g_ in cs["idx"]:
                    cs["hw"][:, cs["idx"][g_]], cs["m"][:, cs["idx"][g_]] = hs > as_, hs - as_

    ref_ew = None
    if tg_ref is not None:
        tg_ref = tg_ref.sort_values("date")
        r_team, r_opp = tg_ref.team_id.map(pos).to_numpy(), tg_ref.opponent_id.map(pos).to_numpy()
        r_fixed = (hca * tg_ref.venue.to_numpy(float)
                   + sum((pre["c"][k] * tg_ref[f"{k}_own"].to_numpy(float) for k in pre["c"]), np.zeros(len(tg_ref)))
                   - sum((pre["c"][k] * tg_ref[f"{k}_opp"].to_numpy(float) for k in pre["c"]), np.zeros(len(tg_ref))))
        r_days = ((tg_ref.date - cutoff).dt.days.to_numpy(float)).clip(min=0)
        ref_ew = np.zeros((n_paths, T))

    events = set(np.unique(days).tolist())
    if cs is not None:
        events |= set(cs["days"].values())
    if tg_ref is not None:
        events |= set(np.unique(r_days).tolist())

    for day in sorted(events):
        if day > last_day:
            walk += rng.normal(0.0, 1.0, (n_paths, T)) * math.sqrt(hz["q_per_day"] * (day - last_day))
            last_day = day
        L = latent(day)
        sel = np.flatnonzero(days == day)
        if len(sel):
            m = (L[:, hi[sel]] - L[:, ai[sel]] + fixed[sel] + rng.normal(0.0, sigma, (n_paths, len(sel)))) * to_pts
            hw = lg.record(hi[sel][None, :], ai[sel][None, :], m)
            if cs is not None:
                for j, g_ in enumerate(rem_ids[sel]):
                    if g_ in cs["idx"]:
                        cs["hw"][:, cs["idx"][g_]], cs["m"][:, cs["idx"][g_]] = hw[:, j], m[:, j]
        if cs is not None:
            _cup_day(day, cs, cup, L, play, lg, pos, rng, rows_p, hca, sigma, to_pts)
        if ref_ew is not None:
            rs = np.flatnonzero(r_days == day)
            if len(rs):
                mu = -L[:, r_opp[rs]] + r_fixed[rs]
                np.add.at(ref_ew.T, r_team[rs], norm.cdf(mu / sigma).T)
    out = {"teams": teams, "wins": lg.wins}
    if conference is not None:
        out["seeds"] = seed_paths(lg, teams, conference, division, rng, latent(last_day), sigma, hca)
    if cs is not None:
        out["cup"] = cs["out"]
    if ref_ew is not None:
        out["d_sos_paths"] = np.bincount(r_team, minlength=T).astype(float) / 2 - ref_ew
    return out


def _cup_day(day, cs, cup, L, play, lg, pos, rng, rows_p, hca, sigma, to_pts) -> None:
    n_paths = len(rows_p)
    ko, days, out = cs["ko"], cs["days"], cs["out"]
    if day == days["qf"]:
        seeds = np.zeros((n_paths, 2, 4), dtype=int)
        for p in range(n_paths):
            qs = CUP.quarterfinal_seeds(cup, CUP.group_tables(cup, cs["hw"][p], cs["m"][p], rng), rng)
            for ci, conf in enumerate(("East", "West")):
                seeds[p, ci] = [pos[t] for t in qs[conf]]
        ko["seeds"] = seeds
        for ci in range(2):
            for j in range(4):
                out["qf"][rows_p, seeds[:, ci, j]] = True
            ko[("w14", ci)], ko[("l14", ci)] = play(L, seeds[:, ci, 0], seeds[:, ci, 3], 1.0)
            ko[("w23", ci)], ko[("l23", ci)] = play(L, seeds[:, ci, 1], seeds[:, ci, 2], 1.0)
            out["sf"][rows_p, ko[("w14", ci)]] = True
            out["sf"][rows_p, ko[("w23", ci)]] = True
    if day == days["sf"] and "seeds" in ko:
        for ci in range(2):
            a_, b_ = ko[("w14", ci)], ko[("w23", ci)]
            sa = np.argmax(ko["seeds"][:, ci, :] == a_[:, None], axis=1)
            sb = np.argmax(ko["seeds"][:, ci, :] == b_[:, None], axis=1)
            a_home = sa < sb                                    # the better (lower) seed hosts
            h_, w_ = np.where(a_home, a_, b_), np.where(a_home, b_, a_)
            win, _ = play(L, h_, w_, 1.0 if cup.sf_home_court else 0.0)
            out["final"][rows_p, win] = True
    if day == days["x0"] and "seeds" in ko:
        for p in range(n_paths):
            qf_teams = set(ko["seeds"][p].ravel().tolist())
            for conf in ("East", "West"):
                pool = [pos[t] for n_, ts in cup.groups.items() if cup.conference[n_] == conf for t in ts
                        if pos[t] not in qf_teams]
                for x_, y_ in CUP.pair_extra(pool, 2, rng):
                    h_, a_ = (x_, y_) if rng.random() < 0.5 else (y_, x_)
                    lg.record_one(p, h_, a_, (L[p, h_] - L[p, a_] + hca + rng.normal(0.0, sigma)) * to_pts)
    if day == days["x1"] and "seeds" in ko:
        for ci in range(2):
            flip = rng.random(n_paths) < 0.5
            a_, b_ = ko[("l14", ci)], ko[("l23", ci)]
            play(L, np.where(flip, a_, b_), np.where(flip, b_, a_), 1.0)


def seed_paths(lg: _League, teams: np.ndarray, conference: pd.Series, division: pd.Series | None, rng,
               L_end: np.ndarray, sigma: float, hca: float) -> np.ndarray:
    """Final seed per path and team: 1-6 by the official standings order, 7-8 from the
    play-in, 9-10 its losers, 11-15 by standings."""
    n_paths, T = lg.wins.shape
    conf = conference.reindex(teams).to_numpy()
    div = division.reindex(teams).to_numpy() if division is not None else conf
    out = np.zeros((n_paths, T), dtype=int)
    order = {c: np.zeros((n_paths, int((conf == c).sum())), dtype=int) for c in np.unique(conf)}
    for p in range(n_paths):
        S = Season(lg.wins[p], lg.W[p], lg.pdiff[p], conf, div, rng)
        for c in order:
            o = S.conference_order(c)
            order[c][p] = o
            out[p, o] = np.arange(1, len(o) + 1)
    r = np.arange(n_paths)

    def game(h, a):
        m = L_end[r, h] - L_end[r, a] + hca + rng.normal(0.0, sigma, n_paths)
        return np.where(m > 0, h, a), np.where(m > 0, a, h)

    for _c, o in order.items():
        w78, l78 = game(o[:, 6], o[:, 7])
        w910, l910 = game(o[:, 8], o[:, 9])
        w_last, l_last = game(l78, w910)
        out[r, w78], out[r, w_last], out[r, l_last], out[r, l910] = 7, 8, 9, 10
    return out


def summarise(sim: dict, wins_now: pd.Series, games_now: pd.Series) -> pd.DataFrame:
    w = sim["wins"]
    t = sim["teams"]
    q = np.percentile(w, [5, 25, 50, 75, 95], axis=0)
    out = pd.DataFrame({"team_id": t, "wins_now": wins_now.reindex(t).fillna(0).to_numpy(),
                        "games_now": games_now.reindex(t).fillna(0).to_numpy(),
                        "exp_wins": w.mean(axis=0), "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]})
    out["exp_remaining_wins"] = out.exp_wins - out.wins_now
    if "seeds" in sim:
        sd = sim["seeds"]
        out["p_top6"] = (sd <= 6).mean(axis=0)
        out["p_playin_or_better"] = (sd <= 10).mean(axis=0)
        out["p_playoffs"] = (sd <= 8).mean(axis=0)
        out["p_seed1"] = (sd == 1).mean(axis=0)
        out["exp_seed"] = sd.mean(axis=0)
    if "cup" in sim:
        for k, v in sim["cup"].items():
            out[f"p_cup_{k}"] = v.mean(axis=0)
    if "d_sos_paths" in sim:
        d = sim["d_sos_paths"]
        out["d_sos_p05"], out["d_sos_p95"] = np.percentile(d, 5, axis=0), np.percentile(d, 95, axis=0)
        rk = (-d).argsort(axis=1).argsort(axis=1) + 1
        out["d_sos_rank_p05"], out["d_sos_rank_p95"] = np.percentile(rk, 5, axis=0), np.percentile(rk, 95, axis=0)
    return out
