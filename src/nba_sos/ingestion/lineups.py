"""Five-man lineup totals per team-season (stats.nba.com LeagueDashLineups filtered by
team). The league-wide call is capped at 2,000 lineups (~60-70% of minutes), so lineups
are fetched team by team, which returns every lineup."""
from __future__ import annotations

import logging
import time

import pandas as pd

from . import nba_stats
from .raw_store import RawStore

log = logging.getLogger(__name__)


def raw_relpath(season: str, team_id: int) -> str:
    return f"nba_stats/team_lineups/{season}/{team_id}.json"


def lineup_client(season: str, team_id: int, season_type: str, timeout: int) -> tuple[str, str]:
    from nba_api.stats.endpoints import leaguedashlineups
    ep = leaguedashlineups.LeagueDashLineups(
        season=season, team_id_nullable=team_id, group_quantity=5, measure_type_detailed_defense="Base",
        per_mode_detailed="Totals", season_type_all_star=season_type, timeout=timeout)
    return ep.nba_response.get_response(), ep.nba_response.get_url()


def fetch_team_lineups(store: RawStore, season: str, team_ids: list[int], cfg: dict,
                       client=lineup_client, sleep=time.sleep) -> int:
    n = 0
    for tid in team_ids:
        rel = raw_relpath(season, tid)
        if store.exists(rel):
            continue
        for attempt in range(1, int(cfg["max_retries"]) + 1):
            try:
                text, url = client(season, tid, cfg["season_type"], int(cfg["request_timeout_s"]))
                nba_stats.parse_payload("team_lineups", text)
                store.write(rel, text.encode("utf-8"), {"source": "stats.nba.com", "endpoint": "leaguedashlineups",
                                                        "name": "team_lineups", "season": season,
                                                        "team_id": int(tid), "url": url, "attempt": attempt})
                n += 1
                break
            except Exception as err:  # noqa: BLE001 - retried, then raised
                if attempt == int(cfg["max_retries"]):
                    raise RuntimeError(f"lineups {season} {tid} failed") from err
                sleep(float(cfg["retry_backoff_s"]) * attempt)
        sleep(float(cfg["sleep_between_requests_s"]))
    return n


def load_team_lineups(store: RawStore, season: str) -> pd.DataFrame:
    frames = []
    for path in sorted((store.root / "nba_stats" / "team_lineups" / season).glob("*.json")):
        frames.append(nba_stats.parse_payload("team_lineups", path.read_text(encoding="utf-8")))
    if not frames:
        raise FileNotFoundError(f"no lineup payloads for {season}; run the fetch step")
    d = pd.concat(frames, ignore_index=True)
    d["season"] = season
    return d
