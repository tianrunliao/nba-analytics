"""Rosters, player impact and team health indices from actual participation.

Roster stints: a player belongs to a team from his first appearance for it until the day
before his first appearance for another team (trade) or the end of the regular season.
A roster game in which he logged no minutes counts as a missed game — this captures
season-ending injuries, which a "between appearances" rule would miss.

Player value (per season, regular season only): minutes-shrunk on/off differential per
48 minutes, from box-score plus/minus:

    on48  = sum(PM) / sum(MIN) * 48
    off48 = sum(team margin - PM) / sum(game minutes - MIN) * 48   (missed games count fully as off)
    value = (on48 - off48) * MIN_on / (MIN_on + K)

Health index of a team-game (players missing, f = player's minutes per game / 48):

    hi_minutes = sum f            hi_value = sum f * value

Both are converted to a per-100 effect by the regular-season team model
(:mod:`nba_sos.ratings.team_strength`), so no hand-set weights are involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHRINK_MINUTES = 1000.0


def team_game_frame(games: pd.DataFrame) -> pd.DataFrame:
    """season, game_id, game_date, team_id, team_margin, game_minutes (48 + 5 per OT)."""
    ot = games.overtime_periods.astype(float).fillna(0)
    gm = 48 + 5 * ot
    h = pd.DataFrame(
        {
            "season": games.season,
            "game_id": games.game_id,
            "game_date": games.game_date,
            "team_id": games.home_team_id,
            "team_margin": games.margin.astype(float),
            "game_minutes": gm,
        }
    )
    a = h.assign(team_id=games.away_team_id, team_margin=-games.margin.astype(float))
    out = pd.concat([h, a], ignore_index=True)
    out["team_id"] = out.team_id.astype("Int64")
    return out


def roster_games(player_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, team roster game): played flag, minutes, plus_minus."""
    tg = team_game_frame(games)
    pg = player_games[player_games.minutes > 0].copy()
    pg["game_date"] = pd.to_datetime(pg.game_date)
    tg["game_date"] = pd.to_datetime(tg.game_date)
    season_end = tg.groupby("season").game_date.max()
    first = pg.groupby(["season", "player_id", "team_id"]).game_date.min().rename("start").reset_index()
    first = first.sort_values(["season", "player_id", "start"])
    first["next_start"] = first.groupby(["season", "player_id"]).start.shift(-1)
    first["end"] = (first.next_start - pd.to_timedelta(1, unit="D")).fillna(first.season.map(season_end))
    rg = first.merge(tg, on=["season", "team_id"])
    rg = rg[(rg.game_date >= rg.start) & (rg.game_date <= rg.end)]
    rg = rg.merge(pg[["game_id", "player_id", "minutes", "plus_minus"]], on=["game_id", "player_id"], how="left")
    rg["played"] = rg.minutes.notna()
    rg["minutes"] = rg.minutes.fillna(0.0)
    rg["plus_minus"] = rg.plus_minus.fillna(0.0)
    # stint is the last one of the season for this player (still on the roster at season end)
    rg["stint_to_season_end"] = rg.next_start.isna()
    return rg.drop(columns=["next_start"])


def player_values(rg: pd.DataFrame, shrink_minutes: float = SHRINK_MINUTES) -> pd.DataFrame:
    """Per (season, player, team stint): mpg, on/off per 48 and shrunk value."""
    x = rg.assign(off_min=rg.game_minutes - rg.minutes, off_pm=rg.team_margin - rg.plus_minus)
    g = x.groupby(["season", "player_id", "team_id"])
    v = pd.DataFrame(
        {
            "games_played": g.played.sum(),
            "roster_games": g.size(),
            "min_on": g.minutes.sum(),
            "pm_on": g.plus_minus.sum(),
            "min_off": g.off_min.sum(),
            "pm_off": g.off_pm.sum(),
        }
    ).reset_index()
    v["mpg"] = (v.min_on / v.games_played.replace(0, np.nan)).fillna(0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        on48 = v.pm_on / v.min_on * 48
        off48 = v.pm_off / v.min_off * 48
    v["on_off48"] = (on48 - off48).fillna(0.0)
    v["value"] = v.on_off48 * v.min_on / (v.min_on + shrink_minutes)
    return v


def health_index(rg: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: hi_minutes and hi_value over roster players who did not play."""
    m = rg[~rg.played].merge(
        values[["season", "player_id", "team_id", "mpg", "value"]], on=["season", "player_id", "team_id"]
    )
    m["f"] = m.mpg / 48
    m["fv"] = m.f * m.value
    out = (
        m.groupby(["game_id", "team_id"])
        .agg(hi_minutes=("f", "sum"), hi_value=("fv", "sum"), n_missing=("f", "size"))
        .reset_index()
    )
    return out
