"""Which players play better in the playoffs than in the regular season?

Box value on one scale for both parts of a season. Official season-level player stats
(stats.nba.com LeagueDashPlayerStats, Advanced + Per100Possessions) are loaded for the
regular season and, with the same definitions, for the playoffs. Both are z-scored with
the regular-season norms of that season (minutes-weighted, players with >= 250 RS
minutes; exactly the norms of :func:`player_value.standardise`) and valued with that
season's lineup-calibrated box weights (``reports/player_values_meta.json``, fit on
seasons <= s), so the regular-season value equals ``pv_box`` and

    delta_raw = beta_s . (z_PO - z_RS - shift_s)

where ``shift_s`` is the playoff-minutes-weighted mean change of each feature among that
season's playoff players: tougher defences and shorter rotations change everyone's
numbers, and only the change relative to the typical playoff player is credited.

Opponent adjustment: delta is regressed (weighted by playoff minutes) on the
minutes-weighted regular-season strength of the playoff opponents faced (schedule-adjusted
net rating per 100, centred within season) and that part is removed.

Noise: a playoff run is 150-1,000 minutes, so most of delta is noise. With
delta_i = mu_player + e_i, Var(e_i) = sigma^2 / m_i (m_i = playoff minutes):
  tau^2   persistent between-player variance, from cross-products of the same player's
          deltas in different postseasons (mean zero if nothing persists)
  sigma^2 from E[delta_i^2] = tau^2 + sigma^2 / m_i
Career estimate (posterior mean) = sum m_i delta_i / (k + sum m_i), k = sigma^2 / tau^2.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import Paths
from ..ingestion import nba_stats
from ..ingestion.raw_store import RawStore
from ..preprocessing.games import PLAYOFF_PREFIX
from . import player_value as V

BOX = V.BOX_FEATURES


def season_norms(rs: pd.DataFrame) -> dict:
    """Regular-season reference: league TS, and minutes-weighted mean / sd of every feature."""
    ref = rs.min_total >= V.MIN_FOR_NORMS
    w = rs.min_total[ref].astype(float)
    lg_ts = float(np.average(rs.ts[ref], weights=w))
    rs = rs.assign(ts_x_usg=(rs.ts - lg_ts) * rs.usg)
    mean = {f: float(np.average(rs[f][ref].astype(float), weights=w)) for f in BOX}
    sd = {f: float(np.sqrt(np.average((rs[f][ref].astype(float) - mean[f]) ** 2, weights=w))) for f in BOX}
    return {"lg_ts": lg_ts, "mean": mean, "sd": sd}


def zscore(d: pd.DataFrame, norms: dict) -> np.ndarray:
    d = d.assign(ts_x_usg=(d.ts - norms["lg_ts"]) * d.usg)
    return np.column_stack([(d[f].astype(float) - norms["mean"][f]) / norms["sd"][f] for f in BOX])


def season_deltas(rs: pd.DataFrame, po: pd.DataFrame, beta: dict) -> pd.DataFrame:
    """Per playoff player of one season: RS and PO box value on the RS scale, and the
    change relative to the playoff-minutes-weighted mean change of each feature."""
    norms = season_norms(rs)
    b = np.array([beta[f] for f in BOX])
    m = po.merge(rs, on="player_id", suffixes=("_po", "_rs"))
    cols = [f for f in BOX if f != "ts_x_usg"] + ["ts"]  # raw inputs of the box features
    zr = zscore(m[[f"{c}_rs" for c in cols]].set_axis(cols, axis=1), norms)
    zp = zscore(m[[f"{c}_po" for c in cols]].set_axis(cols, axis=1), norms)
    dz = zp - zr
    w = m.min_total_po.to_numpy(float)
    shift = np.average(dz, axis=0, weights=w)
    out = pd.DataFrame(
        {
            "player_id": m.player_id,
            "player_name": m.player_name_rs,
            "rs_min": m.min_total_rs,
            "po_min": m.min_total_po,
            "rs_box": zr @ b,
            "po_box": zp @ b,
            "delta_raw": (dz - shift) @ b,
        }
    )
    for c in ("pts_100", "usg", "ts"):
        out[f"d_{c}"] = m[f"{c}_po"].astype(float) - m[f"{c}_rs"].astype(float)
    return out


def opponent_strength(po_logs: pd.DataFrame, pgames: pd.DataFrame, theta: pd.DataFrame) -> pd.DataFrame:
    """Playoff-minutes-weighted regular-season strength of the opponents a player faced."""
    x = po_logs[po_logs.MIN.astype(float) > 0].copy()
    x["GAME_ID"] = x.GAME_ID.astype(str).str.zfill(10)
    x = x[x.GAME_ID.str[:3] == PLAYOFF_PREFIX]
    g = pgames.set_index("game_id")
    home = x.GAME_ID.map(g.home_team_id).astype(int)
    away = x.GAME_ID.map(g.away_team_id).astype(int)
    opp = np.where(x.TEAM_ID.astype(int) == home, away, home)
    th = theta.set_index("team_id").theta
    w = x.MIN.astype(float)
    s = pd.DataFrame({"player_id": x.PLAYER_ID.astype(int), "w": w * pd.Series(opp, index=x.index).map(th), "m": w})
    s = s.groupby("player_id")[["w", "m"]].sum()
    return (s.w / s.m).rename("opp_theta").reset_index()


def opponent_adjust(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Remove the part of delta explained by opponent strength (centred within season)."""
    w = d.po_min.astype(float)
    centre = (d.opp_theta * w).groupby(d.season).transform("sum") / w.groupby(d.season).transform("sum")
    xc = d.opp_theta - centre
    slope = float(np.sum(w * xc * d.delta_raw) / np.sum(w * xc * xc))
    return d.assign(delta=d.delta_raw - slope * xc), slope


def noise_model(d: pd.DataFrame) -> dict:
    """tau^2 from cross-season products of the same player; sigma^2 from E[d^2] - tau^2 ~ 1/m."""
    prods, wts, halves = [], [], []
    for _, g in d.sort_values("season").groupby("player_id"):
        if len(g) < 2:
            continue
        v, m = g.delta.to_numpy(float), g.po_min.to_numpy(float)
        i, j = np.triu_indices(len(g), 1)
        prods.append(v[i] * v[j])
        wts.append(np.sqrt(m[i] * m[j]))
        a, b = g.iloc[::2], g.iloc[1::2]  # split-half: odd- vs even-numbered postseasons
        halves.append(
            (
                np.average(a.delta, weights=a.po_min),
                np.average(b.delta, weights=b.po_min),
                a.po_min.sum(),
                b.po_min.sum(),
            )
        )
    prods, wts = np.concatenate(prods), np.concatenate(wts)
    tau2 = float(np.average(prods, weights=wts))
    tau2_se = float(np.sqrt(np.average((prods - tau2) ** 2, weights=wts) / len(prods)))
    inv_m = 1.0 / d.po_min.to_numpy(float)
    sigma2 = float(np.sum(inv_m * (d.delta.to_numpy(float) ** 2 - tau2)) / np.sum(inv_m**2))
    h = pd.DataFrame(halves, columns=["a", "b", "ma", "mb"])
    big = h[(h.ma >= 300) & (h.mb >= 300)]
    return {
        "tau2": tau2,
        "tau2_se": tau2_se,
        "sigma2": sigma2,
        "k_minutes": sigma2 / tau2 if tau2 > 0 else float("inf"),
        "pairs": int(len(prods)),
        "split_half_corr": float(big.a.corr(big.b)),
        "split_half_players": int(len(big)),
    }


def _wavg(g: pd.DataFrame, col: str) -> float:
    return float(np.average(g[col], weights=g.po_min))


def career(d: pd.DataFrame, noise: dict, min_minutes: float) -> pd.DataFrame:
    """Per player: shrunk career playoff delta, its posterior sd and P(delta > 0)."""
    k, tau2, sigma2 = noise["k_minutes"], noise["tau2"], noise["sigma2"]
    rows = []
    for pid, g in d.groupby("player_id"):
        m = float(g.po_min.sum())
        raw = _wavg(g, "delta")
        rows.append(
            {
                "player_id": pid,
                "player_name": g.sort_values("season").player_name.iloc[-1],
                "postseasons": len(g),
                "first": g.season.min(),
                "last": g.season.max(),
                "po_min": m,
                "rs_box": _wavg(g, "rs_box"),
                "raw_delta": raw,
                "d_pts_100": _wavg(g, "d_pts_100"),
                "d_ts": _wavg(g, "d_ts"),
                "d_usg": _wavg(g, "d_usg"),
                "est": raw * m / (k + m),
                "est_sd": float(np.sqrt(1.0 / (1.0 / tau2 + m / sigma2))),
            }
        )
    c = pd.DataFrame(rows)
    c["p_positive"] = norm.cdf(c.est / c.est_sd)
    return c[c.po_min >= min_minutes].sort_values("est", ascending=False).reset_index(drop=True)


def run(paths: Paths, min_career_minutes: float = 800.0, min_season_minutes: float = 150.0) -> dict:
    from . import players as P
    from . import team_strength as T

    store = RawStore(paths.raw)
    r = lambda n: pd.read_parquet(paths.processed / f"{n}.parquet")  # noqa: E731
    games, pgames = r("games"), r("playoff_games")
    meta = json.loads((paths.reports / "player_values_meta.json").read_text())
    rg = P.roster_games(r("player_games"), games)
    gf = T.game_frame(games, r("team_game_features"), P.health_index(rg, P.player_values(rg)))
    theta = T.fit_pooled(gf, use_health=False, heteroscedastic=True)["theta"]
    rows = []
    for s in sorted(pgames.season.unique()):
        m = season_deltas(
            V.season_player_stats(store, s), V.season_player_stats(store, s, "playoff_"), meta[s]["box_beta"]
        )
        m = m.merge(
            opponent_strength(
                nba_stats.load(store, "playoff_player_game_logs", s),
                pgames[pgames.season == s],
                theta[theta.season == s],
            ),
            on="player_id",
            how="left",
        )
        m["season"] = s
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    d = d[d.po_min >= min_season_minutes].reset_index(drop=True)
    d, slope = opponent_adjust(d)
    nm = noise_model(d)
    nm.update(
        {
            "opponent_slope": slope,
            "player_seasons": int(len(d)),
            "min_season_minutes": min_season_minutes,
            "min_career_minutes": min_career_minutes,
        }
    )
    d["est"] = d.delta * d.po_min / (nm["k_minutes"] + d.po_min)
    return {"seasons": d, "career": career(d, nm, min_career_minutes), "noise": nm}
