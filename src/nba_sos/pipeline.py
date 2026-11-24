"""Command-line entry point.

    python -m nba_sos.pipeline fetch     # network: nba_api payloads + current schedule snapshot
    python -m nba_sos.pipeline fetch-injuries   # network: official injury report PDFs
    python -m nba_sos.pipeline build     # offline: raw -> processed tables, SOS, baseline, audit
    python -m nba_sos.pipeline snapshot  # network: dated current-season schedule + results
    python -m nba_sos.pipeline window-sos --n 15   # current season, first-n-games difficulty
    python -m nba_sos.pipeline playoff-rank        # rolling playoff evaluation + pre-playoff rankings
    python -m nba_sos.pipeline replay              # remaining-season simulation: historical replays (acceptance test)
    python -m nba_sos.pipeline project             # current season: expected wins, seeds, remaining D-SOS
    python -m nba_sos.pipeline playoff-risers      # players who play better in the playoffs than in the regular season
    python -m nba_sos.pipeline analysis            # player values, model variants, star vs depth, top players
    python -m nba_sos.pipeline game-model          # Elo / EWMA / dynamic strength / CatBoost rolling backtest
    python -m nba_sos.pipeline availability        # availability, lineup contribution, injury-module evaluation
"""
from __future__ import annotations

import argparse
import logging

from .config import PROJECT_ROOT, Paths, load_configs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nba_sos.pipeline")
    ap.add_argument("command", choices=["fetch", "fetch-injuries", "build", "snapshot", "window-sos", "playoff-rank", "analysis", "game-model", "availability", "playoff-improve", "playoff-risers", "replay", "project"])
    ap.add_argument("--refresh", action="store_true", help="re-download cached payloads")
    ap.add_argument("--n", type=int, default=15, help="window length for window-sos")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = Paths(PROJECT_ROOT)
    cfg = load_configs(paths)

    if args.command == "fetch":
        from .ingestion.fetch_all import fetch_stats, snapshot_current_schedule
        fetch_stats(paths, cfg, refresh=args.refresh)
        snapshot_current_schedule(paths, cfg)
    elif args.command == "fetch-injuries":
        from .ingestion.injury_reports import fetch_for_seasons
        fetch_for_seasons(paths, cfg)
    elif args.command == "build":
        from .build import build_all
        build_all(paths, cfg)
    elif args.command == "snapshot":
        from .ingestion.fetch_all import snapshot_current_season
        print(snapshot_current_season(paths, cfg))
    elif args.command == "window-sos":
        import json

        from .sos.window_report import run
        out, params = run(paths, cfg, n=args.n)
        season = cfg["data_sources"]["seasons"]["current"]
        dest = paths.reports / f"window_sos_{season}_first{args.n}.csv"
        out.to_csv(dest, index=False)
        (dest.with_suffix(".params.json")).write_text(json.dumps(params, indent=2))
        print(dest)
    elif args.command == "playoff-rank":
        import json

        from .ratings.playoff_report import run as run_po
        metrics, rankings, params = run_po(paths, cfg)
        metrics.to_csv(paths.reports / "playoff_backtest.csv", index=False)
        for season, t in rankings.items():
            t.to_csv(paths.reports / f"playoff_strength_{season}.csv", index=False)
        (paths.reports / "playoff_model_params.json").write_text(json.dumps(params, indent=2, default=float))
        print(metrics.to_string())
    elif args.command == "analysis":
        from .ratings.analysis import run as run_analysis
        out = run_analysis(paths, cfg)
        print(out["variants"].to_string())
        print(out["top"].to_string())
    elif args.command == "game-model":
        import json

        from .models import game_model as GM
        x, tuned = GM.build_features(paths)
        x.to_parquet(paths.processed / "game_model_features.parquet", index=False)
        hist = cfg["data_sources"]["seasons"]["historical"]
        tests = [s for s in hist if s > "2015-16"]
        m, preds, imp = GM.backtest(x, tests)
        m.to_csv(paths.reports / "game_model_backtest.csv", index=False)
        preds.to_parquet(paths.processed / "game_model_predictions.parquet", index=False)
        models = ["preseason", "elo", "ewma14", "dynamic", "cb_strength", "cb_+rest", "cb_+rest+travel", "cb_full", "cb_full_cal"]
        summary = {"all": GM.pooled(m, models), "2022_26": GM.pooled(m[m.season >= "2022-23"], models)}
        for k, v in summary.items():
            v.to_csv(paths.reports / f"game_model_summary_{k}.csv", index=False)
            print(k)
            print(v.round(4).to_string(index=False))
        (paths.reports / "game_model_params.json").write_text(json.dumps({"tuned": tuned, "importance": imp},
                                                                         indent=2, default=float))
    elif args.command == "availability":
        import json

        from .models import availability as AM
        out = AM.run(paths, cfg)
        out["availability"].to_csv(paths.reports / "availability_model.csv", index=False)
        out["evaluation"].to_csv(paths.reports / "availability_injury_evaluation.csv", index=False)
        out["survival"].to_csv(paths.reports / "availability_return_time.csv", index=False)
        out["reasons"].to_csv(paths.reports / "availability_status_by_reason.csv", index=False)
        out["lineup"].to_parquet(paths.processed / "lineup_contribution.parquet", index=False)
        (paths.reports / "availability_params.json").write_text(json.dumps({"info": out["info"], "coefs": out["coefs"]},
                                                                           indent=2, default=float))
        print(out["availability"].round(4).to_string())
        print(out["evaluation"].round(4).to_string())
    elif args.command == "playoff-improve":
        import json

        from .ratings import playoff_improve as PI
        out = PI.run(paths, cfg)
        out["metrics"].to_csv(paths.reports / "playoff_improve_backtest.csv", index=False)
        (paths.reports / "playoff_improve_params.json").write_text(json.dumps(
            {"selected_per_year": out["selected"], "final": out["final"]}, indent=2, default=float))
        series = PI.series_eval(out["blocks"], cfg["data_sources"]["seasons"]["historical"], PI.SERIES_VARIANTS)
        series.to_csv(paths.reports / "playoff_improve_series.csv", index=False)
        names =["base"] + [f"base+{g}" for g in PI.GROUPS] + ["selected"]
        print(PI.pooled(out["metrics"], names).round(4).to_string())
        print(PI.pooled(out["metrics"], names, later=True).round(4).to_string())
        print(json.dumps(out["selected"], indent=1))
        print(json.dumps(out["final"], indent=1, default=float))

    elif args.command == "playoff-risers":
        import json

        from .ratings import playoff_risers as PR
        out = PR.run(paths)
        out["career"].to_csv(paths.reports / "playoff_risers_career.csv", index=False)
        out["seasons"].sort_values("delta", ascending=False).to_csv(
            paths.reports / "playoff_risers_seasons.csv", index=False)
        (paths.reports / "playoff_risers_params.json").write_text(json.dumps(out["noise"], indent=2, default=float))
        print(json.dumps(out["noise"], indent=1, default=float))
        print(out["career"].head(25).round(2).to_string())

    elif args.command == "replay":
        import json

        from .models import season_simulation as SIM
        out = SIM.run_backtest(paths, cfg)
        out["replays"].to_csv(paths.reports / "season_sim_replays.csv", index=False)
        out["metrics"].to_csv(paths.reports / "season_sim_backtest.csv", index=False)
        (paths.reports / "season_sim_params.json").write_text(json.dumps(
            {"per_test_season": out["params"], "sos_check": out["sos_check"]}, indent=2, default=str))
        print(out["metrics"].round(3).to_string())
        print(json.dumps(out["sos_check"], indent=1))
    elif args.command == "project":
        import json

        from .models import season_simulation as SIM
        out = SIM.run_current(paths, cfg)
        season = out["meta"]["season"]
        out["table"].to_csv(paths.reports / f"season_sim_projection_{season}.csv", index=False)
        (paths.reports / f"season_sim_projection_{season}.json").write_text(json.dumps(out["meta"], indent=2, default=str))
        cols = ["team", "conference", "wins_now", "games_now", "exp_wins", "p05", "p95", "p_playoffs", "d_sos", "d_sos_rank"]
        print(out["table"][cols].round(2).to_string())


if __name__ == "__main__":
    main()
