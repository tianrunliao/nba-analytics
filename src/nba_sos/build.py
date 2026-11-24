"""Offline build: raw store -> processed tables, validation, static SOS, baseline, audit.

Writes everything under data/processed/ and reports/, then fails if any validation
check with severity 'error' did not pass (the audit report is written first so the
failure can be inspected)."""
from __future__ import annotations

import logging

import pandas as pd

from .config import Paths
from .features import matchups
from .features.pregame import build_team_game_features
from .ingestion.raw_store import RawStore
from .ingestion.venues import load_venue_config
from .models import preseason
from .preprocessing import availability as av
from .preprocessing.games import PLAYOFF_PREFIX, build_games, build_schedule, excluded_game_counts, load_standings
from .preprocessing.players import build_player_games, team_minutes
from .preprocessing.playoffs import first_playoff_games
from .reporting.audit import write_audit
from .sos.static import static_sos
from .validation import checks as C

log = logging.getLogger(__name__)


class ValidationFailed(RuntimeError):
    pass


def _fetched_at(store: RawStore, name: str, season: str) -> str | None:
    rec = store.latest_records().get(f"nba_stats/{name}/{season}.json")
    return rec["fetched_at"] if rec else None


def build_all(paths: Paths, cfg: dict) -> dict:
    store = RawStore(paths.raw)
    aliases, venues = load_venue_config(paths.configs)
    seasons_cfg = cfg["data_sources"]["seasons"]
    prior, historical = seasons_cfg["prior"], seasons_cfg["historical"]
    results: list[C.CheckResult] = []

    all_games, schedules, players, excluded = [], [], [], {}
    for season in prior + historical:
        g = build_games(store, season, aliases, venues, _fetched_at(store, "league_game_log", season))
        st = load_standings(store, season)
        sc = build_schedule(g, st, venues)
        results += C.check_games(g, season) + C.check_schedule(sc, season) + C.check_local_time(g, sc, season)
        excluded[season] = excluded_game_counts(store, season)
        all_games.append(g)
        if season in historical:
            schedules.append(sc)
            pg = build_player_games(store, season)
            results += C.check_player_minutes(pg, team_minutes(store, season), season)
            players.append(pg)
    all_games = pd.concat(all_games, ignore_index=True)
    games = all_games[all_games.season.isin(historical)].reset_index(drop=True)
    schedule = pd.concat(schedules, ignore_index=True)
    player_games = pd.concat(players, ignore_index=True)

    # injuries
    sel = av.select_reports(games, store, cfg["data_sources"]["injury_reports"])
    parsed, parse_info = av.parse_reports(sel.report_path.dropna().unique().tolist(), store,
                                          paths.interim / "injury_reports_parsed.parquet")
    availability, team_status = av.build_availability(games, sel, parsed, player_games)

    features = build_team_game_features(games, venues, availability, team_status, player_games, cfg["features"])

    # playoffs: games, player participation, snapshot before each team's first playoff game
    po_games, po_players = [], []
    for season in historical:
        pgm = build_games(store, season, aliases, venues, _fetched_at(store, "playoff_game_log", season), kind="playoffs")
        results += C.check_playoff_games(pgm, season)
        po_games.append(pgm)
        po_players.append(build_player_games(store, season, "playoff_player_game_logs", PLAYOFF_PREFIX))
    playoff_games = pd.concat(po_games, ignore_index=True)
    playoff_player_games = pd.concat(po_players, ignore_index=True)
    firsts = pd.concat([first_playoff_games(store, s).assign(season=s)
                        for s in cfg["data_sources"]["injury_reports"]["playoff_start_seasons"]], ignore_index=True)
    first_games = playoff_games[playoff_games.game_id.isin(firsts.game_id)]
    po_sel = av.select_reports(first_games, store, cfg["data_sources"]["injury_reports"])
    po_parsed, po_parse_info = av.parse_reports(po_sel.report_path.dropna().unique().tolist(), store,
                                                paths.interim / "injury_reports_parsed.parquet")
    po_availability, po_team_status = av.build_availability(first_games, po_sel, po_parsed, player_games)
    counts = matchups.matchup_counts(schedule)
    deviations = matchups.structure_deviations(counts)
    sos, sos_per_game = static_sos(games)
    pred, bt = preseason.rolling_backtest(games, all_games, cfg["model"], historical)

    # leakage guard on the produced table (not only in tests)
    leak = features[features.as_of_timestamp.notna() & (features.as_of_timestamp >= features.tipoff_utc)]
    results.append(C._res("features_as_of_before_tipoff", "all", len(leak) > 0,
                          "every as_of_timestamp < tipoff", f"{len(leak)} rows with as_of >= tipoff"))
    results.append(C._res("matchup_matrix_symmetric", "all", matchups.symmetry_violations(counts) > 0,
                          "N_ij == N_ji", "asymmetric meeting counts"))

    out = paths.processed
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "games": games.drop(columns=["_pair", "_in_schedule"]),
        "prior_season_games": all_games[all_games.season.isin(prior)].drop(columns=["_pair", "_in_schedule"]),
        "schedule": schedule, "player_games": player_games,
        "injury_report_selection": sel, "player_availability": availability,
        "injury_team_status": team_status, "team_game_features": features,
        "matchup_counts": counts, "static_sos": sos, "static_sos_per_game": sos_per_game,
        "baseline_predictions": pred,
        "playoff_games": playoff_games.drop(columns=["_pair", "_in_schedule"]),
        "playoff_player_games": playoff_player_games,
        "playoff_start_report_selection": po_sel,
        "playoff_start_availability": po_availability,
        "playoff_start_team_status": po_team_status,
    }
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    bt.to_csv(paths.reports / "baseline_backtest.csv", index=False)
    sos.to_csv(paths.reports / "static_sos.csv", index=False)
    checks = C.apply_known_exceptions(C.results_frame(results), cfg["data_sources"].get("known_exceptions", {}))
    checks.to_csv(paths.reports / "validation_results.csv", index=False)

    write_audit(paths, cfg, store=store, games=games, all_games=all_games, schedule=schedule,
                checks=checks, excluded=excluded, venues=venues, aliases=aliases, counts=counts,
                deviations=deviations, selection=sel, parse_info=parse_info, parsed=parsed,
                availability=availability, team_status=team_status, player_games=player_games,
                features=features, sos=sos, backtest=bt, predictions=pred)

    failed = checks[(checks.severity == "error") & ~checks.passed]
    if len(failed):
        raise ValidationFailed(f"{len(failed)} validation errors:\n{failed.to_string()}")
    log.info("build complete: %d games, %d feature rows", len(games), len(features))
    return tables
