"""Raw nba_api payloads -> ``games`` (one row per game) and ``schedule`` (one row per
team per game). Regular season only (game id prefix 002); NBA Cup final (006),
play-in and playoffs are excluded and counted in the audit instead."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from ..ingestion import nba_stats
from ..ingestion.raw_store import RawStore

REGULAR_SEASON_PREFIX = "002"
PLAYOFF_PREFIX = "004"

# kind -> (game log payload, advanced payload, game id prefix)
KINDS = {
    "regular": ("league_game_log", "team_advanced", REGULAR_SEASON_PREFIX),
    "playoffs": ("playoff_game_log", "playoff_team_advanced", PLAYOFF_PREFIX),
}


def played_mask(lgl: pd.DataFrame) -> pd.Series:
    """Rows of games that were actually played. A cancelled game (BOS-IND, 2013-04-16) is
    listed in LeagueGameLog with no W/L and 0 minutes."""
    return lgl.WL.notna() & (lgl.MIN.astype(float) > 0)


def _gid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(10)


def load_schedule_raw(store: RawStore, season: str) -> pd.DataFrame:
    d = nba_stats.load(store, "schedule", season)
    d["gameId"] = _gid(d.gameId)
    return d


def build_schedule_games(store: RawStore, season: str, prefix: str = REGULAR_SEASON_PREFIX) -> pd.DataFrame:
    """Games of one type (by id prefix) as listed by ScheduleLeagueV2 (final, realised schedule)."""
    d = load_schedule_raw(store, season)
    d = d[d.gameId.str[:3] == prefix]
    out = pd.DataFrame({
        "game_id": d.gameId,
        "season": season,
        "sched_game_date": pd.to_datetime(d.gameDateEst.str[:10]).dt.date,
        "tipoff_utc": pd.to_datetime(d.gameDateTimeUTC, utc=True),
        "sched_home_team_id": d.homeTeam_teamId.astype("Int64"),
        "sched_away_team_id": d.awayTeam_teamId.astype("Int64"),
        "arena_name": d.arenaName,
        "arena_city": d.arenaCity,
        "source_is_neutral": d.isNeutral.astype(bool),
        "game_label": d.gameLabel.fillna(""),
        "game_sublabel": d.gameSubLabel.fillna(""),
        "game_subtype": d.gameSubtype.fillna(""),
        "game_status": d.gameStatusText,
        "home_team_local_time": d.homeTeamTime,
    })
    return out.reset_index(drop=True)


def excluded_game_counts(store: RawStore, season: str) -> dict[str, int]:
    """Counts of non-regular-season games in the schedule, by id prefix."""
    d = load_schedule_raw(store, season)
    return d.gameId.str[:3].value_counts().sort_index().to_dict()


def _team_rows(store: RawStore, season: str, kind: str = "regular") -> pd.DataFrame:
    lgl_name, adv_name, prefix = KINDS[kind]
    lgl = nba_stats.load(store, lgl_name, season)
    adv = nba_stats.load(store, adv_name, season)
    lgl["GAME_ID"], adv["GAME_ID"] = _gid(lgl.GAME_ID), _gid(adv.GAME_ID)
    lgl = lgl[lgl.GAME_ID.str[:3] == prefix]
    lgl = lgl[played_mask(lgl)]
    t = lgl.merge(adv[["GAME_ID", "TEAM_ID", "POSS"]], on=["GAME_ID", "TEAM_ID"], how="left",
                  validate="one_to_one")
    # Home/away comes from the schedule's designated home team: for neutral-site games
    # (2024-25 onward) LeagueGameLog writes "@" in both teams' MATCHUP strings.
    sched = build_schedule_games(store, season, prefix)[["game_id", "sched_home_team_id"]]
    t = t.merge(sched, left_on="GAME_ID", right_on="game_id", how="left")
    t["is_home"] = t.TEAM_ID == t.sched_home_team_id
    t["matchup_says_home"] = t.MATCHUP.str.contains(" vs. ", regex=False)
    return t.drop(columns=["game_id", "sched_home_team_id"])


def build_games(store: RawStore, season: str, aliases: pd.DataFrame, venues: pd.DataFrame,
                fetched_at: str | None = None, kind: str = "regular") -> pd.DataFrame:
    t = _team_rows(store, season, kind)
    keep = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL", "POSS",
            "PLUS_MINUS", "MIN", "matchup_says_home"]
    h = t[t.is_home][keep].add_prefix("h_")
    a = t[~t.is_home][keep].add_prefix("a_")
    g = h.merge(a, left_on="h_GAME_ID", right_on="a_GAME_ID", how="outer", indicator=True)
    sched = build_schedule_games(store, season, KINDS[kind][2])
    g = g.rename(columns={"h_GAME_ID": "game_id"})
    g["game_id"] = g.game_id.fillna(g.a_GAME_ID)
    g = g.merge(sched, on="game_id", how="left", indicator="_sched")

    games = pd.DataFrame({
        "game_id": g.game_id,
        "season": season,
        "game_date": pd.to_datetime(g.h_GAME_DATE.fillna(g.a_GAME_DATE)).dt.date,
        "sched_game_date": g.sched_game_date,
        "tipoff_utc": g.tipoff_utc,
        "home_team_id": g.h_TEAM_ID.astype("Int64"),
        "away_team_id": g.a_TEAM_ID.astype("Int64"),
        "sched_home_team_id": g.sched_home_team_id,
        "sched_away_team_id": g.sched_away_team_id,
        "home_team": g.h_TEAM_ABBREVIATION,
        "away_team": g.a_TEAM_ABBREVIATION,
        "home_score": g.h_PTS.astype("Int64"),
        "away_score": g.a_PTS.astype("Int64"),
        "home_wl": g.h_WL,
        "away_wl": g.a_WL,
        "home_plus_minus": g.h_PLUS_MINUS,
        # team minutes: 240 regulation, +25 per overtime period (5 players x 5 min)
        "overtime_periods": ((g.h_MIN - 240) / 25).astype("Int64"),
        "home_matchup_says_home": g.h_matchup_says_home.astype("boolean"),
        "away_matchup_says_home": g.a_matchup_says_home.astype("boolean"),
        "home_poss_source": g.h_POSS,
        "away_poss_source": g.a_POSS,
        "arena_name": g.arena_name,
        "arena_city": g.arena_city,
        "source_is_neutral": g.source_is_neutral.astype("boolean"),
        "game_label": g.game_label,
        "game_sublabel": g.game_sublabel,
        "game_subtype": g.game_subtype,
        "game_status": g.game_status,
        "home_team_local_time": g.home_team_local_time,
        "_pair": g["_merge"].astype(str),
        "_in_schedule": g["_sched"].astype(str),
    })
    games["home_win"] = (games.home_score > games.away_score).astype("Int64")
    games["margin"] = (games.home_score - games.away_score).astype("Int64")
    # Two teams' POSS for one game can differ by a few; use their mean (decision #3).
    games["possessions"] = (games.home_poss_source + games.away_poss_source) / 2
    games["home_off_rating"] = games.home_score / games.possessions * 100
    games["away_off_rating"] = games.away_score / games.possessions * 100
    games["net_rating_margin"] = games.home_off_rating - games.away_off_rating

    al = aliases[["arena_name", "arena_city", "venue_id"]]
    games = games.merge(al, on=["arena_name", "arena_city"], how="left")
    v = venues[["venue_id", "neutral_site"]].rename(columns={"neutral_site": "venue_neutral_site"})
    games = games.merge(v, on="venue_id", how="left")
    games["is_neutral"] = games.source_is_neutral.fillna(False).astype(bool) | \
        games.venue_neutral_site.fillna(False).astype(bool)
    games["source_fetched_at"] = fetched_at
    return games.sort_values(["tipoff_utc", "game_id"]).reset_index(drop=True)


def load_standings(store: RawStore, season: str) -> pd.DataFrame:
    s = nba_stats.load(store, "standings", season)
    return pd.DataFrame({"season": season, "team_id": s.TeamID.astype("Int64"),
                         "conference": s.Conference, "division": s.Division})


def build_schedule(games: pd.DataFrame, standings: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game (pre-game knowable fields only; no results)."""
    base = ["season", "game_id", "game_date", "tipoff_utc", "is_neutral", "venue_id"]
    home = games[base + ["home_team_id", "home_team", "away_team_id", "away_team"]].rename(columns={
        "home_team_id": "team_id", "home_team": "team", "away_team_id": "opponent_id",
        "away_team": "opponent"})
    home["home_away"] = "H"
    away = games[base + ["away_team_id", "away_team", "home_team_id", "home_team"]].rename(columns={
        "away_team_id": "team_id", "away_team": "team", "home_team_id": "opponent_id",
        "home_team": "opponent"})
    away["home_away"] = "A"
    s = pd.concat([home, away], ignore_index=True)
    st = standings[["season", "team_id", "conference", "division"]]
    s = s.merge(st, on=["season", "team_id"], how="left")
    s = s.merge(st.rename(columns={"team_id": "opponent_id", "conference": "opponent_conference",
                                   "division": "opponent_division"}),
                on=["season", "opponent_id"], how="left")
    v = venues[["venue_id", "latitude", "longitude", "timezone"]].rename(columns={
        "latitude": "arena_latitude", "longitude": "arena_longitude", "timezone": "venue_timezone"})
    s = s.merge(v, on="venue_id", how="left")
    s["tipoff_local_time"] = [
        t.tz_convert(ZoneInfo(z)).strftime("%Y-%m-%dT%H:%M") if pd.notna(t) and isinstance(z, str) else None
        for t, z in zip(s.tipoff_utc, s.venue_timezone)]
    return s.sort_values(["season", "team_id", "tipoff_utc"]).reset_index(drop=True)
