"""Per player-season values (regular season) and the regular-season top-player tables.

For season s every input is the regular season of s (complete before its playoffs);
the box-score weights are fit on team-seasons <= s only, so the values can feed the
playoff model of season s without look-ahead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import season_start_year
from ..ingestion.lineups import load_team_lineups
from ..ingestion.raw_store import RawStore
from . import lineup_rapm as L
from . import player_value as V
from . import players as P

log = logging.getLogger(__name__)


def compute_player_values(
    store: RawStore, games: pd.DataFrame, features: pd.DataFrame, player_games: pd.DataFrame, seasons: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Per player-season:
    pv_box  box-score prior: era-standardised box features weighted by coefficients fit
            on five-man lineup results of seasons <= s only
    lpm     lineup RAPM shrunk toward kappa * pv_box (kappa, lambda by CV)
    z_onoff box-score on/off z-score (reference only)"""
    rg = P.roster_games(player_games, games)
    vals = P.player_values(rg)
    stats = pd.concat([V.season_player_stats(store, s) for s in seasons], ignore_index=True)
    z = V.standardise(stats, V.onoff_by_season(vals), V.team_net_for_players(games, player_games))
    lineups = {s: load_team_lineups(store, s) for s in seasons}
    pace = {s: float(games[games.season == s].possessions.mean()) for s in seasons}
    # Lineups keep between-team variation (demean=False): within-team-only weights were
    # found to under-credit dominant interior bigs (negative rebounding weight), team-season
    # totals to over-credit them; pooling all lineup variation sits between the two.
    designs = {s: L.lineup_box_design(lineups[s], pace[s], z, V.BOX_FEATURES, demean=False) for s in seasons}
    out, meta = [], {}
    for s in seasons:
        upto = [x for x in seasons if season_start_year(x) <= season_start_year(s)]
        fb = L.fit_box_on_lineups([designs[x] for x in upto], V.BOX_FEATURES)
        beta = np.array([fb["beta"][f] for f in V.BOX_FEATURES])
        zs = z[z.season == s]
        inp = zs[["season", "player_id", "player_name", "min_total", "gp"]].copy()
        inp["pv_box"] = zs[[f"z_{f}" for f in V.BOX_FEATURES]].to_numpy(float) @ beta
        inp["z_onoff"] = zs.z_on_off48.to_numpy(float)
        lp, info = L.fit_season(lineups[s], pace[s], inp.set_index("player_id").pv_box)
        x = inp.merge(lp.drop(columns="prior"), on="player_id", how="left")
        x["lpm"] = x.lpm.fillna(x.pv_box)  # players in no lineup keep their prior
        x["team_abbr"] = x.player_id.map(_main_team(player_games, games, s))
        out.append(x)
        meta[s] = {
            "box_beta": fb["beta"],
            "box_lineups": fb["n_lineups"],
            "lpm_lambda": info["lambda"],
            "lpm_kappa": info["kappa"],
            "lpm_lineups": info["n_lineups"],
        }
        log.info("%s: kappa=%s lambda=%s lineups=%d", s, info["kappa"], info["lambda"], info["n_lineups"])
    return pd.concat(out, ignore_index=True), meta


def _main_team(player_games: pd.DataFrame, games: pd.DataFrame, season: str) -> pd.Series:
    code = dict(zip(games.home_team_id.astype(int), games.home_team))
    m = player_games[player_games.season == season].groupby(["player_id", "team_id"]).minutes.sum().reset_index()
    m = m.sort_values("minutes").drop_duplicates("player_id", keep="last")
    return m.set_index("player_id").team_id.astype(int).map(code)


def min_minutes(season: str, games: pd.DataFrame) -> float:
    """1,500 minutes in an 82-game season, pro-rated by games per team (COVID seasons)."""
    g = games[games.season == season]
    per_team = pd.concat([g.home_team_id, g.away_team_id]).value_counts().median()
    return 1500.0 * per_team / 82


def top_players(pv: pd.DataFrame, games: pd.DataFrame, seasons: list[str], n: int = 5) -> pd.DataFrame:
    rows = []
    for s in seasons:
        x = pv[(pv.season == s) & (pv.min_total >= min_minutes(s, games))].copy()
        x["rank"] = x.lpm.rank(ascending=False, method="first").astype(int)
        x["box_rank"] = x.pv_box.rank(ascending=False, method="min").astype(int)
        x["rapm_rank"] = x.rapm_no_prior.rank(ascending=False, method="min").astype(int)
        rows.append(x.sort_values("rank").head(n))
    cols = [
        "season",
        "rank",
        "player_name",
        "team_abbr",
        "gp",
        "min_total",
        "lpm",
        "pv_box",
        "rapm_no_prior",
        "box_rank",
        "rapm_rank",
    ]
    return pd.concat(rows, ignore_index=True)[cols]


def team_star_depth(roster_mpg: pd.DataFrame, value_col: str = "lpm") -> pd.DataFrame:
    """Per team (season-end roster, RS minutes per game):
    star  = mean value of the two best players among the 8 highest-minute players
    depth = mpg-weighted mean value of the players ranked 6-9 by minutes (second unit)"""
    rows = []
    for tid, g in roster_mpg.groupby("team_id"):
        g = g.sort_values("mpg", ascending=False)
        top8 = g.head(8)
        bench = g.iloc[5:9]
        rows.append(
            {
                "team_id": tid,
                "star": float(top8[value_col].nlargest(2).mean()),
                "depth": float(np.average(bench[value_col], weights=bench.mpg)) if len(bench) else 0.0,
            }
        )
    return pd.DataFrame(rows)
