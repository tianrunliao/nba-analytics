"""Model-variant comparison, star-vs-depth analysis, top-player and team-season ranking tables.

python -m nba_sos.pipeline analysis
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import Paths
from ..ingestion.raw_store import RawStore
from . import playoff_model as PM
from . import playoff_report as R
from .player_report import compute_player_values, top_players

log = logging.getLogger(__name__)
TOP_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
RANKING_FIRST_SEASON = "2013-14"


def pooled(metrics: pd.DataFrame, models=("full", "theta_only", "raw_net", "winpct")) -> dict:
    w, s = metrics.n_games, metrics.series
    out = {"games": int(w.sum()), "series": int(s.sum())}
    for m in models:
        out[f"{m}_log_loss"] = float((metrics[f"{m}_log_loss"] * w).sum() / w.sum())
        out[f"{m}_series_acc"] = float((metrics[f"{m}_series_acc"] * s).sum() / s.sum())
        out[f"{m}_series_brier"] = float((metrics[f"{m}_series_brier"] * s).sum() / s.sum())
    return out


def minutes_concentration(d: dict) -> pd.DataFrame:
    """Share of team minutes by minutes rank (1-2, 3-5, 6-9, 10+): RS vs playoffs."""
    out = []
    for kind, pg in (("regular", d["player_games"]), ("playoffs", d["playoff_player_games"])):
        m = pg.groupby(["season", "team_id", "player_id"]).minutes.sum().reset_index()
        if kind == "playoffs":
            m = m[m.team_id.isin(d["playoff_player_games"].team_id.unique())]
        m["rk"] = m.groupby(["season", "team_id"]).minutes.rank(ascending=False, method="first")
        m["band"] = pd.cut(m.rk, [0, 2, 5, 9, 99], labels=["top1-2", "3-5", "6-9", "10+"])
        tot = m.groupby(["season", "team_id"]).minutes.transform("sum")
        m["share"] = m.minutes / tot
        b = m.groupby(["season", "team_id", "band"], observed=True).share.sum().groupby("band", observed=True).mean()
        out.append(b.rename(kind))
    return pd.concat(out, axis=1)


def run(paths: Paths, cfg: dict) -> dict:
    d = R._load(paths)
    hist = cfg["data_sources"]["seasons"]["historical"]
    store = RawStore(paths.raw)
    pv, meta = compute_player_values(store, d["games"], d["team_game_features"], d["player_games"], hist)
    pv.to_parquet(paths.processed / "player_values.parquet", index=False)
    (paths.reports / "player_values_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    variants = {
        "V0 on/off, per-100": dict(value_source="onoff", heteroscedastic=False, scale=False, feats=None),
        "V1 + noise-weighted seasons": dict(value_source="onoff", heteroscedastic=True, scale=False, feats=None),
        "V2 + per-season noise scaling": dict(value_source="onoff", heteroscedastic=True, scale=True, feats=None),
        "V3 lineup RAPM value": dict(value_source="lpm", heteroscedastic=True, scale=True, feats=None),
        "V4 + star/depth": dict(
            value_source="lpm",
            heteroscedastic=True,
            scale=True,
            feats=PM.FEATS[:-1] + ["d_star", "d_depth", "home_flag"],
        ),
    }
    rows, results = [], {}
    for name, v in variants.items():
        blocks = R.season_blocks(
            paths, cfg, d, pv=pv, value_source=v["value_source"], heteroscedastic=v["heteroscedastic"]
        )
        if v["scale"]:
            blocks = R.scale_blocks(blocks)
        metrics, rankings, params = R.evaluate_and_rank(blocks, hist, v["feats"])
        results[name] = (blocks, metrics, rankings, params)
        p_all, p_recent = pooled(metrics), pooled(metrics[metrics.season >= "2022-23"])
        rows.append(
            {
                "variant": name,
                **{f"all_{k}": val for k, val in p_all.items()},
                **{f"2023_26_{k}": val for k, val in p_recent.items()},
            }
        )
        log.info("%s: %s", name, p_all)
    comp = pd.DataFrame(rows)
    comp.to_csv(paths.reports / "model_variants.csv", index=False)

    # star vs depth: pooled playoff regression over all postseasons (descriptive) + per-year fits
    blocks, metrics, rankings, params = results["V4 + star/depth"]
    tau = 40.0
    allrows = pd.concat([b["rows"][tau] for b in blocks.values()], ignore_index=True)
    feats = PM.FEATS[:-1] + ["d_star", "d_depth", "home_flag"]
    pooled_fit = PM.fit_playoff(allrows, feats)
    sd_star = float(allrows.d_star.std())
    sd_depth = float(allrows.d_depth.std())
    star_depth = {
        "pooled_all_postseasons": {
            "coef": pooled_fit["coef"],
            "se": pooled_fit["se"],
            "n_games": pooled_fit["n_games"],
            "sd_d_star": sd_star,
            "sd_d_depth": sd_depth,
        },
        "per_test_year": {
            y: {k: p[k] for k in ("c_d_star", "se_d_star", "c_d_depth", "se_d_depth")} for y, p in params.items()
        },
        "minutes_concentration": minutes_concentration(d).to_dict(),
    }
    (paths.reports / "star_vs_depth.json").write_text(json.dumps(star_depth, indent=2, default=float))

    tops = top_players(pv, d["games"], TOP_SEASONS)
    tops.to_csv(paths.reports / "top_players_2019_2026.csv", index=False)

    ranks = team_season_rankings(paths, RANKING_FIRST_SEASON)
    ranks.to_csv(paths.reports / "team_season_rankings_2013_2026.csv", index=False)
    return {"variants": comp, "star_depth": star_depth, "top": tops, "rankings": ranks, "results": results}


def team_season_rankings(paths: Paths, first_season: str = RANKING_FIRST_SEASON) -> pd.DataFrame:
    """Every team-season's regular-season strength from ``first_season`` on (the pooled fit
    uses those seasons only), per 100 possessions vs an average team on a neutral court:
      adj_net      schedule-, venue- and back-to-back-adjusted (actual rosters)
      healthy_net  additionally adjusted for missing players (full-roster strength)
      neutral_win  P(beat an average team on a neutral court) using that season's game
                   noise sd — the era-standardised scale (game noise rose ~11% since 2013-14)
    plus wins, raw net rating and the playoff round reached."""
    from scipy.stats import norm

    from ..preprocessing.playoffs import series_table
    from . import players as P
    from . import team_strength as T

    d = R._load(paths)
    for k in ("games", "team_game_features", "player_games", "playoff_games"):
        d[k] = d[k][d[k].season >= first_season]
    g, pg = d["games"], d["player_games"]
    rg = P.roster_games(pg, g)
    gf = T.game_frame(g, d["team_game_features"], P.health_index(rg, P.player_values(rg)))
    adj = T.fit_pooled(gf, use_health=False, heteroscedastic=True)
    hlt = T.fit_pooled(gf, use_health=True, heteroscedastic=True)
    t = (
        adj["theta"]
        .rename(columns={"theta": "adj_net"})
        .merge(hlt["theta"].rename(columns={"theta": "healthy_net"}), on=["season", "team_id"])
    )
    t["neutral_win"] = norm.cdf(t.adj_net / t.season.map(adj["season_sigma"]))
    code = dict(zip(g.home_team_id.astype(int), g.home_team))
    t["team"] = t.team_id.map(code)
    long = pd.concat(
        [
            g[["season", "home_team_id", "home_score", "away_score", "possessions"]].set_axis(
                ["season", "team_id", "pf", "pa", "poss"], axis=1
            ),
            g[["season", "away_team_id", "away_score", "home_score", "possessions"]].set_axis(
                ["season", "team_id", "pf", "pa", "poss"], axis=1
            ),
        ]
    )
    long[["pf", "pa", "poss"]] = long[["pf", "pa", "poss"]].astype(float)
    long["team_id"] = long.team_id.astype(int)
    agg = (
        long.groupby(["season", "team_id"])
        .agg(games=("pf", "size"), wins=("pf", lambda s: 0), pf=("pf", "sum"), pa=("pa", "sum"), poss=("poss", "sum"))
        .reset_index()
    )
    w = long.assign(w=(long.pf > long.pa).astype(int)).groupby(["season", "team_id"]).w.sum()
    agg["wins"] = agg.set_index(["season", "team_id"]).index.map(w)
    agg["raw_net"] = (agg.pf - agg.pa) / agg.poss * 100
    t = t.merge(agg[["season", "team_id", "games", "wins", "raw_net"]], on=["season", "team_id"])
    po = d["playoff_games"].assign(
        home_team_id=lambda x: x.home_team_id.astype(int), away_team_id=lambda x: x.away_team_id.astype(int)
    )
    sr = series_table(po)
    rr = (
        pd.concat(
            [
                sr[["season", "team_lo", "round"]].rename(columns={"team_lo": "team_id"}),
                sr[["season", "team_hi", "round"]].rename(columns={"team_hi": "team_id"}),
            ]
        )
        .groupby(["season", "team_id"])["round"]
        .max()
    )
    champs = set(zip(sr[sr["round"] == 4].season, sr[sr["round"] == 4].winner))
    t["playoff_round"] = [5 if (s, tid) in champs else rr.get((s, tid), 0) for s, tid in zip(t.season, t.team_id)]
    t = t.sort_values("adj_net", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    t["rank_era_std"] = t.neutral_win.rank(ascending=False, method="min").astype(int)
    return t
