"""Player game logs -> ``player_games`` (minutes per player per game)."""

from __future__ import annotations

import pandas as pd

from ..ingestion import nba_stats
from ..ingestion.injury_reports import norm_key
from ..ingestion.raw_store import RawStore
from .games import REGULAR_SEASON_PREFIX, played_mask


def build_player_games(
    store: RawStore, season: str, name: str = "player_game_logs", prefix: str = REGULAR_SEASON_PREFIX
) -> pd.DataFrame:
    p = nba_stats.load(store, name, season)
    p["GAME_ID"] = p.GAME_ID.astype(str).str.zfill(10)
    p = p[p.GAME_ID.str[:3] == prefix]
    return pd.DataFrame(
        {
            "season": season,
            "game_id": p.GAME_ID,
            "game_date": pd.to_datetime(p.GAME_DATE).dt.date,
            "player_id": p.PLAYER_ID.astype("Int64"),
            "player_name": p.PLAYER_NAME,
            "player_key": p.PLAYER_NAME.map(norm_key),
            "team_id": p.TEAM_ID.astype("Int64"),
            "minutes": p.MIN.astype(float),
            "plus_minus": p.PLUS_MINUS.astype(float),
        }
    ).reset_index(drop=True)


def team_minutes(store: RawStore, season: str) -> pd.DataFrame:
    lgl = nba_stats.load(store, "league_game_log", season)
    lgl["GAME_ID"] = lgl.GAME_ID.astype(str).str.zfill(10)
    lgl = lgl[(lgl.GAME_ID.str[:3] == REGULAR_SEASON_PREFIX) & played_mask(lgl)]
    return pd.DataFrame(
        {"game_id": lgl.GAME_ID, "team_id": lgl.TEAM_ID.astype("Int64"), "team_min": lgl.MIN.astype(float)}
    )
