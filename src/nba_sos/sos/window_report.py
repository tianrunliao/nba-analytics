"""First-n-games schedule difficulty for the current season, from the latest snapshot.

    python -m nba_sos.pipeline window-sos --n 15
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from ..features.pregame import back_to_back_context, schedule_density, team_game_long, travel
from ..ingestion.raw_store import RawStore
from ..ingestion.venues import load_venue_config
from ..preprocessing.games import build_games, build_schedule_games, load_schedule_raw
from . import window

log = logging.getLogger(__name__)


def latest_snapshot(paths: Paths) -> Path:
    snaps = sorted((paths.raw / "current").glob("*/manifest.jsonl"))
    if not snaps:
        raise FileNotFoundError("no current-season snapshot; run `python -m nba_sos.pipeline snapshot`")
    return snaps[-1].parent


def season_games_frame(store: RawStore, season: str, aliases: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """All scheduled regular-season games with known teams; results where completed."""
    sched = build_schedule_games(store, season)
    # NBA Cup knockout games whose teams are not yet known carry team id 0
    known = sched.sched_home_team_id.fillna(0).ne(0) & sched.sched_away_team_id.fillna(0).ne(0)
    sched = sched[known]
    done = build_games(store, season, aliases, venues)
    res_cols = ["game_id", "home_score", "away_score", "possessions", "overtime_periods", "home_team", "away_team"]
    g = sched.merge(done[res_cols], on="game_id", how="left")
    g = g.merge(aliases[["arena_name", "arena_city", "venue_id"]], on=["arena_name", "arena_city"], how="left")
    missing = g[g.venue_id.isna()][["arena_name", "arena_city"]].drop_duplicates()
    if len(missing):
        raise ValueError(f"games at venues missing from configs/venue_aliases.csv: {missing.to_dict('records')}")
    g = g.merge(venues[["venue_id", "neutral_site"]], on="venue_id", how="left")
    g["is_neutral"] = g.source_is_neutral | g.neutral_site.astype("boolean").fillna(False).astype(bool)
    g["home_team_id"], g["away_team_id"] = g.sched_home_team_id, g.sched_away_team_id
    g["game_date"] = g.sched_game_date
    g["completed"] = g.home_score.notna()
    return g


def tricodes(store: RawStore, season: str) -> dict:
    raw = load_schedule_raw(store, season)
    m = dict(zip(raw.homeTeam_teamId, raw.homeTeam_teamTricode))
    m.update(dict(zip(raw.awayTeam_teamId, raw.awayTeam_teamTricode)))
    return {int(k): v for k, v in m.items() if pd.notna(k) and isinstance(v, str)}


def run(paths: Paths, cfg: dict, n: int = 15) -> tuple[pd.DataFrame, dict]:
    season = cfg["data_sources"]["seasons"]["current"]
    snap = latest_snapshot(paths)
    store = RawStore(snap)
    aliases, venues = load_venue_config(paths.configs)
    g = season_games_frame(store, season, aliases, venues)
    code = tricodes(store, season)
    g["home_team"], g["away_team"] = g.home_team_id.map(code), g.away_team_id.map(code)

    # history (processed tables from `build`)
    hist = pd.read_parquet(paths.processed / "games.parquet")
    prior_only = pd.read_parquet(paths.processed / "prior_season_games.parquet")
    all_hist = pd.concat([prior_only, hist], ignore_index=True)
    done = g[g.completed]
    gp = pd.concat([done.home_team_id, done.away_team_id]).value_counts()
    n_fit = int(gp.median())
    params = window.fit_strength(hist, all_hist, n_fit)
    params.update({"n_fit": n_fit, "snapshot": snap.name, "completed_games": int(len(done)),
                   "games_played_min": int(gp.min()), "games_played_max": int(gp.max())})

    # strengths now: prior (last season) + this season to date
    prior = pd.read_parquet(paths.processed / "games.parquet")
    pr = window.preseason.prior_ratings(prior)
    pr = pr[pr.season == season][["team_id", "prior_net_rating"]]
    tl = team_game_long(done.assign(possessions=done.possessions.astype(float)))
    cur = tl.groupby("team_id").apply(
        lambda d: (d.pts_for.sum() - d.pts_against.sum()) / d.possessions.sum() * 100,
        include_groups=False).rename("net_to_date").reset_index()
    st = pr.merge(cur, on="team_id", how="outer")
    st["team_id"] = st.team_id.astype("Int64")
    st["S"] = params["a"] * st.prior_net_rating + params["c"] * st.net_to_date
    st["team"] = st.team_id.map(code)

    # window = each team's first n scheduled games
    t = team_game_long(g.assign(home_score=g.home_score.astype("Float64"), away_score=g.away_score.astype("Float64"),
                                possessions=g.possessions.astype(float)))
    t = pd.concat([t, schedule_density(t, [4, 7]), travel(t, venues), back_to_back_context(t, venues)], axis=1)
    t["k"] = t.groupby(["season", "team_id"]).cumcount()
    opp = t[["game_id", "team_id", "rest_days", "is_back_to_back"]].rename(
        columns={"team_id": "opponent_id", "rest_days": "opp_rest_days", "is_back_to_back": "opp_b2b"})
    t = t.merge(opp, on=["game_id", "opponent_id"], how="left")
    t["team_id"], t["opponent_id"] = t.team_id.astype("Int64"), t.opponent_id.astype("Int64")
    w = t[t.k < n].copy()

    # opponent strength excluding the opponent's games against this team (OWP convention)
    tot = tl.groupby("team_id")[["pts_for", "pts_against", "possessions"]].sum()
    h2h = tl.groupby(["team_id", "opponent_id"])[["pts_for", "pts_against", "possessions"]].sum()
    pairs = w[["team_id", "opponent_id"]].drop_duplicates()
    rows = []
    for i, j in zip(pairs.team_id, pairs.opponent_id):
        t_j = tot.loc[j] - (h2h.loc[(j, i)] if (j, i) in h2h.index else 0)
        net_ex = (t_j.pts_for - t_j.pts_against) / t_j.possessions * 100 if t_j.possessions > 0 else np.nan
        prior_j = st.set_index("team_id").prior_net_rating.get(j, np.nan)
        rows.append({"team_id": i, "opponent_id": j, "S": params["a"] * prior_j + params["c"] * net_ex})
    pair_s = pd.DataFrame(rows)
    pair_s["team_id"], pair_s["opponent_id"] = pair_s.team_id.astype("Int64"), pair_s.opponent_id.astype("Int64")
    out = window.window_sos(w, pair_s, params)
    burden = w.groupby("team_id").agg(
        played=("pts_for", lambda s: int(s.notna().sum())),
        home=("home_away", lambda s: int((s == "H").sum())),
        b2b=("is_back_to_back", lambda s: int(s.fillna(False).sum())),
        b2b_road_road=("b2b_type", lambda s: int((s == "AA").sum())),
        opp_on_b2b=("opp_b2b", lambda s: int(s.fillna(False).sum())),
        rest_edge=("rest_days", "sum"), opp_rest=("opp_rest_days", "sum"),
        travel_km=("travel_distance_km", "sum"), max_road_run=("consecutive_road_games", "max")).reset_index()
    burden["rest_edge"] = burden.rest_edge - burden.opp_rest
    out = out.merge(burden.drop(columns=["opp_rest"]), on="team_id").merge(
        st[["team_id", "prior_net_rating", "net_to_date", "S"]], on="team_id")
    return out.sort_values("d_sos_rank").reset_index(drop=True), params
