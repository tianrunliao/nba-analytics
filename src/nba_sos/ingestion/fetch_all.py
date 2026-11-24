"""Network step: download every raw payload Phase 1 needs."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Paths
from . import nba_stats
from .raw_store import RawStore

log = logging.getLogger(__name__)

# Which endpoints each season group needs.
PRIOR_ENDPOINTS = ["league_game_log", "team_advanced", "schedule", "standings"]
HISTORICAL_ENDPOINTS = ["league_game_log", "team_advanced", "player_game_logs", "schedule", "standings"]
CURRENT_ENDPOINTS = ["schedule"]
PLAYOFF_ENDPOINTS = ["playoff_game_log", "playoff_team_advanced", "playoff_player_game_logs"]
PLAYER_SEASON_ENDPOINTS = ["player_advanced", "player_per100", "playoff_player_advanced", "playoff_player_per100"]


def fetch_stats(paths: Paths, cfg: dict, *, refresh: bool = False) -> list[dict]:
    store = RawStore(paths.raw)
    seasons = cfg["data_sources"]["seasons"]
    stats_cfg = cfg["data_sources"]["nba_stats"]
    jobs = [(n, s) for s in seasons["prior"] for n in PRIOR_ENDPOINTS]
    jobs += [
        (n, s)
        for s in seasons["historical"]
        for n in HISTORICAL_ENDPOINTS + PLAYOFF_ENDPOINTS + PLAYER_SEASON_ENDPOINTS
    ]
    records = []
    for name, season in jobs:
        rec = nba_stats.fetch(store, name, season, stats_cfg, refresh=refresh)
        log.info("%s %s -> %s", name, season, "cached" if rec is None else rec["path"])
        if rec:
            records.append(rec)
    from . import lineups

    for season in seasons["historical"]:
        lgl = nba_stats.load(store, "league_game_log", season)
        n = lineups.fetch_team_lineups(store, season, sorted(lgl.TEAM_ID.astype(int).unique()), stats_cfg)
        log.info("team_lineups %s -> %d new", season, n)
    return records


def snapshot_current_schedule(paths: Paths, cfg: dict) -> dict:
    """Store a dated snapshot of the current season's schedule. Snapshots are
    time-sensitive (they cannot be re-fetched later), so each run adds a new file."""
    from .raw_store import utc_now_iso

    store = RawStore(paths.raw)
    season = cfg["data_sources"]["seasons"]["current"]
    stats_cfg = cfg["data_sources"]["nba_stats"]
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    text, url = nba_stats.nba_api_client(
        "schedule", season, stats_cfg["season_type"], int(stats_cfg["request_timeout_s"])
    )
    nba_stats.parse_payload("schedule", text)
    return store.write(
        f"nba_stats/schedule_snapshots/{season}/{stamp}.json",
        text.encode("utf-8"),
        {
            "source": "stats.nba.com",
            "endpoint": "scheduleleaguev2",
            "name": "schedule_snapshot",
            "season": season,
            "url": url,
        },
    )


CURRENT_SNAPSHOT_ENDPOINTS = ["schedule", "league_game_log", "team_advanced", "player_game_logs"]


def snapshot_current_season(paths: Paths, cfg: dict) -> Path:
    """Dated snapshot of the current season (schedule + results so far). Each snapshot is
    its own RawStore root under data/raw/current/<stamp>/, so every existing loader works
    on it unchanged and earlier snapshots are never overwritten."""
    from .raw_store import utc_now_iso

    season = cfg["data_sources"]["seasons"]["current"]
    stamp = utc_now_iso().replace(":", "").replace("-", "").replace("+0000", "Z")
    root = paths.raw / "current" / stamp
    store = RawStore(root)
    for name in CURRENT_SNAPSHOT_ENDPOINTS:
        nba_stats.fetch(store, name, season, cfg["data_sources"]["nba_stats"])
    return root
