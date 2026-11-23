"""Fetch season-level bulk payloads from stats.nba.com via nba_api.

Raw response text is stored verbatim; parsing back to DataFrames reuses nba_api's own
response parser so offline rebuilds match what nba_api would have produced.
"""
from __future__ import annotations

import time
from typing import Callable

import pandas as pd
from nba_api.stats.library.http import NBAStatsResponse

from .raw_store import RawStore

# name -> (nba_api endpoint name, builder(season, season_type, timeout) -> endpoint object)
def _league_game_log(season, season_type, timeout, get_request=True):
    from nba_api.stats.endpoints import leaguegamelog
    return leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star=season_type,
        player_or_team_abbreviation="T", timeout=timeout, get_request=get_request)


def _team_advanced(season, season_type, timeout, get_request=True):
    from nba_api.stats.endpoints import teamgamelogs
    return teamgamelogs.TeamGameLogs(
        season_nullable=season, season_type_nullable=season_type,
        measure_type_player_game_logs_nullable="Advanced", timeout=timeout,
        get_request=get_request)


def _player_game_logs(season, season_type, timeout, get_request=True):
    from nba_api.stats.endpoints import playergamelogs
    return playergamelogs.PlayerGameLogs(
        season_nullable=season, season_type_nullable=season_type, timeout=timeout,
        get_request=get_request)


def _player_advanced(season, season_type, timeout, get_request=True):
    from nba_api.stats.endpoints import leaguedashplayerstats
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, season_type_all_star=season_type, measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals", timeout=timeout, get_request=get_request)


def _player_per100(season, season_type, timeout, get_request=True):
    from nba_api.stats.endpoints import leaguedashplayerstats
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, season_type_all_star=season_type, measure_type_detailed_defense="Base",
        per_mode_detailed="Per100Possessions", timeout=timeout, get_request=get_request)


def _schedule(season, season_type, timeout, get_request=True):
    from nba_api.stats.endpoints import scheduleleaguev2
    return scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=timeout,
                                             get_request=get_request)


def _standings(season, season_type, timeout, get_request=True):
    from nba_api.stats.endpoints import leaguestandingsv3
    return leaguestandingsv3.LeagueStandingsV3(
        season=season, season_type=season_type, timeout=timeout, get_request=get_request)


ENDPOINTS: dict[str, tuple[str, Callable]] = {
    "league_game_log": ("leaguegamelog", _league_game_log),
    "team_advanced": ("teamgamelogs", _team_advanced),
    "player_game_logs": ("playergamelogs", _player_game_logs),
    "schedule": ("scheduleleaguev2", _schedule),
    "standings": ("leaguestandingsv3", _standings),
    # playoff payloads: same endpoints, season type forced by PLAYOFF_SEASON_TYPE
    "playoff_game_log": ("leaguegamelog", _league_game_log),
    "playoff_team_advanced": ("teamgamelogs", _team_advanced),
    "playoff_player_game_logs": ("playergamelogs", _player_game_logs),
    # official season-level player stats (regular season)
    "player_advanced": ("leaguedashplayerstats", _player_advanced),
    "player_per100": ("leaguedashplayerstats", _player_per100),
    "playoff_player_advanced": ("leaguedashplayerstats", _player_advanced),
    "playoff_player_per100": ("leaguedashplayerstats", _player_per100),
}

# Names whose season type is fixed regardless of the configured default.
SEASON_TYPE_OVERRIDE = {
    "playoff_game_log": "Playoffs",
    "playoff_team_advanced": "Playoffs",
    "playoff_player_game_logs": "Playoffs",
    "playoff_player_advanced": "Playoffs",
    "playoff_player_per100": "Playoffs",
}

# Data set to extract from each payload.
DATASET_NAME = {
    "league_game_log": "LeagueGameLog",
    "team_advanced": "TeamGameLogs",
    "player_game_logs": "PlayerGameLogs",
    "schedule": "SeasonGames",
    "standings": "Standings",
    "playoff_game_log": "LeagueGameLog",
    "playoff_team_advanced": "TeamGameLogs",
    "playoff_player_game_logs": "PlayerGameLogs",
    "player_advanced": "LeagueDashPlayerStats",
    "player_per100": "LeagueDashPlayerStats",
    "playoff_player_advanced": "LeagueDashPlayerStats",
    "playoff_player_per100": "LeagueDashPlayerStats",
    "team_lineups": "Lineups",
}

# Endpoints whose payload nba_api parses with an endpoint-specific parser.
_CUSTOM_PARSER = {"schedule"}


def raw_relpath(name: str, season: str) -> str:
    return f"nba_stats/{name}/{season}.json"


def nba_api_client(name: str, season: str, season_type: str, timeout: int) -> tuple[str, str]:
    """Perform the real request; return (response_text, url)."""
    _, builder = ENDPOINTS[name]
    ep = builder(season, season_type, timeout)
    return ep.nba_response.get_response(), ep.nba_response.get_url()


def fetch(store: RawStore, name: str, season: str, cfg: dict, *, refresh: bool = False,
          client: Callable[..., tuple[str, str]] = nba_api_client,
          sleep: Callable[[float], None] = time.sleep) -> dict | None:
    """Fetch one endpoint/season into the raw store. Returns the manifest record,
    or None when cached. Raises after ``max_retries`` failed attempts."""
    relpath = raw_relpath(name, season)
    if store.exists(relpath) and not refresh:
        return None
    retries = int(cfg["max_retries"])
    season_type = SEASON_TYPE_OVERRIDE.get(name, cfg["season_type"])
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            text, url = client(name, season, season_type, int(cfg["request_timeout_s"]))
            parse_payload(name, text)  # refuse to store payloads we cannot parse
            rec = store.write(relpath, text.encode("utf-8"), {
                "source": "stats.nba.com", "endpoint": ENDPOINTS[name][0], "name": name,
                "season": season, "season_type": season_type, "url": url,
                "attempt": attempt})
            sleep(float(cfg["sleep_between_requests_s"]))
            return rec
        except Exception as err:  # noqa: BLE001 - re-raised below after retries
            last_err = err
            if attempt < retries:
                sleep(float(cfg["retry_backoff_s"]) * attempt)
    raise RuntimeError(f"fetch {name} {season} failed after {retries} attempts") from last_err


def parse_payload(name: str, text: str) -> pd.DataFrame:
    resp = NBAStatsResponse(response=text, status_code=200, url="")
    endpoint = ENDPOINTS[name][0] if name in _CUSTOM_PARSER else None
    data_sets = resp.get_data_sets(endpoint)
    ds = data_sets[DATASET_NAME[name]]
    return pd.DataFrame(ds["data"], columns=ds["headers"])


def load(store: RawStore, name: str, season: str) -> pd.DataFrame:
    path = store.path_for(raw_relpath(name, season))
    if not path.exists():
        raise FileNotFoundError(f"raw payload missing: {path}; run the fetch step first")
    return parse_payload(name, path.read_text(encoding="utf-8"))
