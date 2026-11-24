"""Playoff games and per-team series context."""

from __future__ import annotations

import pandas as pd

from ..ingestion.raw_store import RawStore
from .games import PLAYOFF_PREFIX, build_schedule_games


def first_playoff_games(store: RawStore, season: str) -> pd.DataFrame:
    """Each playoff team's first playoff game (game_id, tipoff_utc), for injury snapshots."""
    s = build_schedule_games(store, season, PLAYOFF_PREFIX)
    s = s[s.game_status.astype(str).str.startswith("Final")]
    long = pd.concat(
        [
            s[["game_id", "tipoff_utc", "sched_home_team_id"]].rename(columns={"sched_home_team_id": "team_id"}),
            s[["game_id", "tipoff_utc", "sched_away_team_id"]].rename(columns={"sched_away_team_id": "team_id"}),
        ]
    )
    first = long.sort_values("tipoff_utc").drop_duplicates("team_id")
    return first[["game_id", "tipoff_utc"]].drop_duplicates("game_id").reset_index(drop=True)


def series_table(playoff_games: pd.DataFrame) -> pd.DataFrame:
    """One row per series: teams, games, wins, winner, round (1-4 from the game id)."""
    g = playoff_games.copy()
    # NBA playoff ids: 004 YY 00 R S G -> R = round, S = series within round, G = game
    g["round"] = g.game_id.str[7].astype(int)
    g["series_no"] = g.game_id.str[8].astype(int)
    g["team_lo"] = g[["home_team_id", "away_team_id"]].min(axis=1)
    g["team_hi"] = g[["home_team_id", "away_team_id"]].max(axis=1)
    g["lo_win"] = ((g.home_team_id == g.team_lo) == (g.home_score > g.away_score)).astype(int)
    out = (
        g.groupby(["season", "round", "series_no", "team_lo", "team_hi"])
        .agg(games=("game_id", "size"), lo_wins=("lo_win", "sum"), first_tipoff=("tipoff_utc", "min"))
        .reset_index()
    )
    out["hi_wins"] = out.games - out.lo_wins
    out["winner"] = out.team_lo.where(out.lo_wins > out.hi_wins, out.team_hi)
    return out
