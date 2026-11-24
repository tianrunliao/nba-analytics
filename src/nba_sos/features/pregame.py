"""Pre-game team features (``team_game_features``).

Information rule: results-based features for a game on ET date d use only games with
ET date < d (validation guarantees one game per team per date, so this equals "all
earlier games"). Schedule/travel features use the schedule, which is published before
the season. Injury features use only the pre-tip-off report chosen in
:mod:`nba_sos.preprocessing.availability`.
"""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
ASSUMED_MAX_GAME_DURATION = timedelta(hours=4)


def team_game_long(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, from each team's perspective, with results."""
    cols = ["season", "game_id", "game_date", "tipoff_utc", "is_neutral", "venue_id", "possessions", "overtime_periods"]
    h = games[cols].assign(
        team_id=games.home_team_id,
        team=games.home_team,
        opponent_id=games.away_team_id,
        home_away="H",
        pts_for=games.home_score,
        pts_against=games.away_score,
    )
    a = games[cols].assign(
        team_id=games.away_team_id,
        team=games.away_team,
        opponent_id=games.home_team_id,
        home_away="A",
        pts_for=games.away_score,
        pts_against=games.home_score,
    )
    t = pd.concat([h, a], ignore_index=True)
    t["win"] = (t.pts_for > t.pts_against).astype(float)
    t["margin"] = (t.pts_for - t.pts_against).astype(float)
    t["pts_for"], t["pts_against"] = t.pts_for.astype(float), t.pts_against.astype(float)
    return t.sort_values(["season", "team_id", "tipoff_utc"]).reset_index(drop=True)


def efficiency_to_date(t: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date aggregates over strictly earlier games (shifted cumulative sums)."""
    g = t.groupby(["season", "team_id"], sort=False)
    prior = lambda col: g[col].transform(lambda s: s.cumsum().shift(1))  # noqa: E731
    n = g.cumcount().astype(float)
    pf, pa, poss = prior("pts_for"), prior("pts_against"), prior("possessions")
    wins, margin = prior("win"), prior("margin")
    out = pd.DataFrame(index=t.index)
    out["season_games_played"] = n.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["season_win_pct_to_date"] = (wins / n).where(n > 0)
        out["season_off_rating_to_date"] = (pf / poss * 100).where(n > 0)
        out["season_def_rating_to_date"] = (pa / poss * 100).where(n > 0)
        out["season_net_rating_to_date"] = ((pf - pa) / poss * 100).where(n > 0)
        out["season_pace_to_date"] = (poss / n).where(n > 0)
        out["season_avg_margin_to_date"] = (margin / n).where(n > 0)
    return out


def _count_in_window(dates: pd.Series, days: int) -> pd.Series:
    """Games on [d-(days-1), d], including the current game."""
    ords = np.array([d.toordinal() for d in dates])
    lo = np.searchsorted(ords, ords - (days - 1), side="left")
    return pd.Series(np.arange(len(ords)) - lo + 1, index=dates.index)


def schedule_density(t: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=t.index)
    g = t.groupby(["season", "team_id"], sort=False)
    prev_date = g.game_date.shift(1)
    rest = pd.Series(
        [(d - p).days - 1 if pd.notna(p) else np.nan for d, p in zip(t.game_date, prev_date)], index=t.index
    )
    out["rest_days"] = rest
    out["is_back_to_back"] = (rest == 0).astype("boolean").where(rest.notna())
    for w in windows:
        out[f"games_last_{w}_days"] = g.game_date.transform(lambda s, w=w: _count_in_window(s, w))
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def home_venues(t: pd.DataFrame) -> pd.DataFrame:
    """Primary home venue per team-season: most frequent venue of non-neutral home games."""
    h = t[(t.home_away == "H") & ~t.is_neutral]
    c = h.groupby(["season", "team_id", "venue_id"]).size().rename("n").reset_index()
    c = c.sort_values(["season", "team_id", "n", "venue_id"], ascending=[True, True, False, True])
    return c.drop_duplicates(["season", "team_id"])[["season", "team_id", "venue_id"]].rename(
        columns={"venue_id": "home_venue_id"}
    )


def travel(t: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """Distance (great-circle, km) and time-zone shift from the previous game's venue.
    Season opener: previous location = team's primary home venue (decision #9)."""
    v = venues.set_index("venue_id")
    hv = home_venues(t)
    tt = t[["season", "team_id", "venue_id", "tipoff_utc", "home_away", "is_neutral"]].merge(
        hv, on=["season", "team_id"], how="left"
    )
    tt.index = t.index
    prev = t.groupby(["season", "team_id"], sort=False).venue_id.shift(1)
    origin = prev.fillna(tt.home_venue_id)
    out = pd.DataFrame(index=t.index)
    out["prev_venue_id"] = origin
    out["travel_origin"] = np.where(prev.isna(), "season_start_home_venue", "previous_game_venue")
    out["travel_distance_km"] = haversine_km(
        v.latitude.reindex(origin).values,
        v.longitude.reindex(origin).values,
        v.latitude.reindex(t.venue_id).values,
        v.longitude.reindex(t.venue_id).values,
    )

    def offset_h(zone, ts):
        return ts.tz_convert(ZoneInfo(zone)).utcoffset().total_seconds() / 3600

    tz_cur = v.timezone.reindex(t.venue_id).values
    tz_prev = v.timezone.reindex(origin).values
    # both offsets evaluated at the current tip-off instant, so DST changes are not counted as travel
    out["timezone_change_hours"] = [
        offset_h(c, ts) - offset_h(p, ts) for c, p, ts in zip(tz_cur, tz_prev, t.tipoff_utc)
    ]
    # road = away, or neutral site for either team
    road = (t.home_away == "A") | t.is_neutral
    run = road.groupby([t.season, t.team_id]).transform(lambda s: s.groupby((~s).cumsum()).cumsum())
    out["consecutive_road_games"] = run.astype(int)
    return out


def _local_hour(ts: pd.Timestamp, zone) -> float:
    if pd.isna(ts) or not isinstance(zone, str):
        return np.nan
    lt = ts.tz_convert(ZoneInfo(zone))
    return lt.hour + lt.minute / 60


def back_to_back_context(t: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """Context of consecutive games, so back-to-back difficulty can be combined with
    home/away, distance, direction and timing.

    leg          'H' home / 'A' away; neutral sites count as 'A' for both teams.
    b2b_type     previous leg + current leg on back-to-backs ('AA' = both away), else None.
    turnaround_hours        hours between the previous and current tip-offs.
    prev_overtime_periods   overtime played in the previous game (known before this one).
    tipoff_local_hour / prev_tipoff_local_hour   wall-clock hour at each game's venue.
    body_clock_tipoff_hour  current tip-off in the team's home-venue time zone.
    Together with travel_distance_km and timezone_change_hours (+ = eastward) these
    describe e.g. 'late road game in the West, then early road game in the East'.
    """
    tz = venues.set_index("venue_id").timezone
    g = t.groupby(["season", "team_id"], sort=False)
    out = pd.DataFrame(index=t.index)
    leg = pd.Series(np.where((t.home_away == "H") & ~t.is_neutral, "H", "A"), index=t.index)
    prev_leg = leg.groupby([t.season, t.team_id]).shift(1)
    prev_date = g.game_date.shift(1)
    b2b = pd.Series([pd.notna(p) and (d - p).days == 1 for d, p in zip(t.game_date, prev_date)], index=t.index)
    out["leg"] = leg
    out["b2b_type"] = (prev_leg + leg).where(b2b)
    out["turnaround_hours"] = (t.tipoff_utc - g.tipoff_utc.shift(1)).dt.total_seconds() / 3600
    out["prev_overtime_periods"] = g.overtime_periods.shift(1).astype("Float64")
    local = pd.Series([_local_hour(ts, tz.get(v)) for ts, v in zip(t.tipoff_utc, t.venue_id)], index=t.index)
    out["tipoff_local_hour"] = local
    out["prev_tipoff_local_hour"] = local.groupby([t.season, t.team_id]).shift(1)
    home_tz = home_venues(t).assign(home_tz=lambda d: d.home_venue_id.map(tz))
    ht = t[["season", "team_id"]].merge(home_tz, on=["season", "team_id"], how="left").home_tz
    out["body_clock_tipoff_hour"] = [_local_hour(ts, z) for ts, z in zip(t.tipoff_utc, ht)]
    return out


def injury_features(
    t: pd.DataFrame,
    availability: pd.DataFrame,
    team_status: pd.DataFrame,
    player_games: pd.DataFrame,
    missing_statuses: list[str],
) -> pd.DataFrame:
    """Counts by status and the season-to-date average minutes of missing players.
    NaN when no usable pre-tip-off snapshot exists (never imputed as 0)."""
    key = t[["season", "game_id", "team_id", "game_date"]].copy()
    key["team_id"] = key.team_id.astype("Int64")
    ts = team_status[
        ["game_id", "team_id", "snapshot_usable", "not_submitted", "report_found", "report_published_at_utc"]
    ]
    k = key.merge(ts, on=["game_id", "team_id"], how="left")
    # rows whose team could not be resolved are excluded here and counted in the audit
    a = availability[availability.status_valid.astype(bool) & availability.team_id.notna()].copy()

    counts = a.pivot_table(
        index=["game_id", "team_id"], columns="injury_status", values="player_key", aggfunc="count", fill_value=0
    )
    for s in ["Out", "Doubtful", "Questionable", "Probable"]:
        if s not in counts.columns:
            counts[s] = 0
    counts = (
        counts[["Out", "Doubtful", "Questionable", "Probable"]].rename(columns=lambda s: f"n_{s.lower()}").reset_index()
    )

    # season-to-date minutes per game for each player, strictly before the game date
    pg = player_games.sort_values(["season", "player_id", "game_date"])
    gp = pg.groupby(["season", "player_id"])
    pg = pg.assign(cum_min=gp.minutes.cumsum(), cum_n=gp.cumcount() + 1)
    miss = a[a.injury_status.isin(missing_statuses) & a.player_id.notna()].merge(
        key[["game_id", "team_id", "game_date"]], on=["game_id", "team_id"], how="left"
    )
    mins = []
    for (season, pid), grp in miss.groupby(["season", "player_id"]):
        hist = pg[(pg.season == season) & (pg.player_id == pid)]
        hd = np.array([d.toordinal() for d in hist.game_date])
        for idx, d in zip(grp.index, grp.game_date):
            j = np.searchsorted(hd, d.toordinal(), side="left") - 1  # last game with date < d
            mins.append((idx, hist.cum_min.iloc[j] / hist.cum_n.iloc[j] if j >= 0 else 0.0))
    if mins:
        miss.loc[[i for i, _ in mins], "prior_mpg"] = [m for _, m in mins]
    else:
        miss["prior_mpg"] = pd.Series(dtype=float)
    mm = miss.groupby(["game_id", "team_id"]).prior_mpg.sum().rename("missing_minutes_prior_avg").reset_index()
    unmatched = (
        a[a.injury_status.isin(missing_statuses) & a.player_id.isna()]
        .groupby(["game_id", "team_id"])
        .size()
        .rename("n_missing_unmatched")
        .reset_index()
    )

    k = (
        k.merge(counts, on=["game_id", "team_id"], how="left")
        .merge(mm, on=["game_id", "team_id"], how="left")
        .merge(unmatched, on=["game_id", "team_id"], how="left")
    )
    usable = k.snapshot_usable.fillna(False).astype(bool)
    cols = ["n_out", "n_doubtful", "n_questionable", "n_probable", "missing_minutes_prior_avg", "n_missing_unmatched"]
    for c in cols:
        k[c] = k[c].fillna(0).where(usable)
    k["n_missing_players"] = (k.n_out + k.n_doubtful).where(usable)
    k["injury_snapshot_usable"] = usable
    k.index = t.index
    return k[["injury_snapshot_usable", "report_published_at_utc", *cols, "n_missing_players"]].rename(
        columns={"report_published_at_utc": "injury_report_published_at_utc"}
    )


def results_cutoff(t: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """Upper bound on when the latest result used could have been known: the latest
    tip-off among league games on earlier ET dates + an assumed max game duration."""
    by_date = games.groupby("game_date").tipoff_utc.max().sort_index()
    prev_max = by_date.cummax().shift(1)
    return t.game_date.map(prev_max) + ASSUMED_MAX_GAME_DURATION


def build_team_game_features(
    games: pd.DataFrame,
    venues: pd.DataFrame,
    availability: pd.DataFrame,
    team_status: pd.DataFrame,
    player_games: pd.DataFrame,
    feature_cfg: dict,
) -> pd.DataFrame:
    t = team_game_long(games)
    parts = [
        t[
            [
                "season",
                "game_id",
                "game_date",
                "tipoff_utc",
                "team_id",
                "team",
                "opponent_id",
                "home_away",
                "is_neutral",
                "venue_id",
            ]
        ],
        efficiency_to_date(t),
        schedule_density(t, feature_cfg["schedule_density"]["windows_days"]),
        travel(t, venues),
        back_to_back_context(t, venues),
        injury_features(t, availability, team_status, player_games, feature_cfg["injury"]["missing_statuses"]),
    ]
    f = pd.concat(parts, axis=1)
    f["results_cutoff_utc"] = results_cutoff(t, games)  # index-aligned; keeps tz
    f["as_of_timestamp"] = f[["results_cutoff_utc", "injury_report_published_at_utc"]].max(axis=1)
    return f.reset_index(drop=True)
