"""Per-game pre-tip-off injury snapshot -> ``player_availability`` and ``injury_team_status``.

For each game the report used is the latest downloaded slot whose publication time
(HTTP Last-Modified) is strictly before tip-off — the same walk-back the fetch step
performs. Only rows of that game (date + matchup) are kept.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from nba_api.stats.static import teams as static_teams

from ..ingestion import injury_reports as ir
from ..ingestion.raw_store import RawStore


def nickname_keys() -> dict[str, str]:
    return {t["abbreviation"]: ir.norm_key(t["nickname"]) for t in static_teams.get_teams()}


def select_reports(games: pd.DataFrame, store: RawStore, cfg: dict) -> pd.DataFrame:
    new_start = datetime.fromisoformat(cfg["new_start_et"])
    lookback = int(cfg["max_slot_lookback"])
    known = store.latest_records()
    rows = []
    for gid, tip in zip(games.game_id, games.tipoff_utc):
        chosen = None
        if pd.notna(tip):
            tip_et = tip.tz_convert(ir.ET).to_pydatetime()
            for nominal in ir.slot_nominal_times(tip_et, new_start, lookback):
                rec = known.get(ir.raw_relpath(ir.slot_filename(nominal, new_start)))
                if rec and rec.get("published_at_utc") and pd.Timestamp(rec["published_at_utc"]) < tip:
                    chosen = rec
                    break
        rows.append({"game_id": gid, "report_path": chosen["path"] if chosen else None,
                     "report_published_at_utc": pd.Timestamp(chosen["published_at_utc"]) if chosen else pd.NaT,
                     "report_sha256": chosen["sha256"] if chosen else None})
    out = pd.DataFrame(rows)
    out["report_published_at_utc"] = pd.to_datetime(out.report_published_at_utc, utc=True)
    return out


def parse_reports(paths: list[str], store: RawStore, cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse each report once; cache parsed rows keyed by path (raw files are immutable)."""
    cached = pd.read_parquet(cache_path) if cache_path.exists() else None
    cached_info = pd.read_parquet(cache_path.with_name(cache_path.stem + "_info.parquet")) \
        if cache_path.exists() else None
    # files that failed to parse earlier are retried (the parser may have been extended)
    if cached_info is not None:
        cached_info = cached_info[cached_info.parse_error.isna()]
    done = set(cached_info.report_path) if cached_info is not None else set()
    nick = nickname_keys()
    new_rows, new_info = [], []
    for p in sorted(set(paths) - done):
        try:
            df, info = ir.parse_report_pdf(store.path_for(p))
            df = ir.resolve_rows(df, nick)
            df["report_path"] = p
            new_rows.append(df)
            new_info.append({"report_path": p, "parse_error": None,
                             "report_time_et": info["report_time_et"], "pages": info["pages"],
                             "orphan_fragments": info["orphan_fragments"]})
        except Exception as err:  # noqa: BLE001 - recorded in the audit, never silently dropped
            new_info.append({"report_path": p, "parse_error": f"{type(err).__name__}: {err}",
                             "report_time_et": None, "pages": None, "orphan_fragments": None})
    rows = pd.concat([x for x in [cached, *new_rows] if x is not None and len(x)], ignore_index=True) \
        if (new_rows or cached is not None) else pd.DataFrame()
    # built from records, not pd.concat: concatenating the all-NA columns of reports that failed
    # to parse again triggers a pandas FutureWarning and a NumPy generic-unit NaT DeprecationWarning
    info_records = ([] if cached_info is None else cached_info.to_dict("records")) + new_info
    info = pd.DataFrame(info_records) if (new_info or cached_info is not None) else pd.DataFrame()
    if new_rows or new_info:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(cache_path, index=False)
        info.to_parquet(cache_path.with_name(cache_path.stem + "_info.parquet"), index=False)
    return rows, info


def build_availability(games: pd.DataFrame, selected: pd.DataFrame, parsed: pd.DataFrame,
                       player_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (player_availability, injury_team_status)."""
    g = games[["season", "game_id", "game_date", "tipoff_utc", "home_team_id", "away_team_id",
               "home_team", "away_team"]].merge(selected, on="game_id", how="left")
    if len(parsed):
        p = parsed.rename(columns={"game_date": "report_game_date"})
        m = g.merge(p, left_on=["report_path", "game_date", "away_team", "home_team"],
                    right_on=["report_path", "report_game_date", "away_tricode", "home_tricode"], how="inner")
    else:
        m = pd.DataFrame(columns=list(g.columns) + ["team_tricode", "player", "player_key", "status",
                                                    "reason", "not_submitted", "status_valid"])
    m["team_id"] = m.home_team_id.where(m.team_tricode == m.home_team, m.away_team_id.where(m.team_tricode == m.away_team))

    # player-id matching: same season & team first, then unique-in-season name
    roster = player_games.drop_duplicates(["season", "team_id", "player_key", "player_id"])
    by_team = roster.groupby(["season", "team_id", "player_key"]).player_id.agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else pd.NA).reset_index()
    by_season = roster.drop_duplicates(["season", "player_key", "player_id"]).groupby(
        ["season", "player_key"]).player_id.agg(lambda s: s.iloc[0] if s.nunique() == 1 else pd.NA).reset_index()
    players = m[~m.not_submitted.astype(bool)].copy()
    players = players.merge(by_team, on=["season", "team_id", "player_key"], how="left")
    players = players.merge(by_season.rename(columns={"player_id": "pid2"}), on=["season", "player_key"], how="left")
    players["match_method"] = players.player_id.notna().map({True: "season_team", False: None})
    players.loc[players.player_id.isna() & players.pid2.notna(), "match_method"] = "season_unique_name"
    players["player_id"] = players.player_id.fillna(players.pid2).astype("Int64")
    pa = pd.DataFrame({
        "season": players.season, "game_id": players.game_id, "team_id": players.team_id.astype("Int64"),
        "team": players.team_tricode, "player_id": players.player_id, "player_name_raw": players.player,
        "player_key": players.player_key, "injury_status": players.status,
        "status_valid": players.status_valid, "reason": players.reason,
        "report_path": players.report_path, "report_timestamp": players.report_published_at_utc,
        "match_method": players.match_method,
        "source": "official NBA injury report (pre-tip-off snapshot)"})

    # team-level status: one row per team per game
    long = pd.concat([
        g[["season", "game_id", "home_team_id", "home_team", "report_path", "report_published_at_utc"]]
        .rename(columns={"home_team_id": "team_id", "home_team": "team"}),
        g[["season", "game_id", "away_team_id", "away_team", "report_path", "report_published_at_utc"]]
        .rename(columns={"away_team_id": "team_id", "away_team": "team"})], ignore_index=True)
    listed = m.groupby(["game_id", "team_id"]).agg(
        n_rows=("status", "size"), not_submitted=("not_submitted", "max")).reset_index()
    listed["team_id"] = listed.team_id.astype("Int64")
    game_listed = m.groupby("game_id").size().rename("game_rows").reset_index()
    ts = long.merge(listed, on=["game_id", "team_id"], how="left").merge(game_listed, on="game_id", how="left")
    ts["report_found"] = ts.report_path.notna()
    ts["game_in_report"] = ts.game_rows.fillna(0) > 0
    ts["not_submitted"] = ts.not_submitted.astype("boolean").fillna(False).astype(bool)
    ts["n_rows"] = ts.n_rows.fillna(0).astype(int)
    # usable snapshot: report found, game listed, and team did submit
    ts["snapshot_usable"] = ts.report_found & ts.game_in_report & ~ts.not_submitted
    return pa.reset_index(drop=True), ts.drop(columns=["game_rows"]).reset_index(drop=True)
