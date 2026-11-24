"""End-to-end check on the real local raw store (skipped when it has not been fetched)."""

import shutil

import pandas as pd
import pytest

from nba_sos.config import PROJECT_ROOT, Paths, load_configs

PATHS = Paths(PROJECT_ROOT)
pytestmark = pytest.mark.skipif(
    not (PATHS.raw / "manifest.jsonl").exists(), reason="raw data not fetched; run `python -m nba_sos.pipeline fetch`"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from nba_sos.build import build_all

    out = tmp_path_factory.mktemp("build")

    class TmpPaths(Paths):
        processed = property(lambda self: out / "processed")
        reports = property(lambda self: out / "reports")
        interim = property(lambda self: out / "interim")

    p = TmpPaths(PROJECT_ROOT)
    p.reports.mkdir(parents=True)
    # reuse the parsed-PDF cache (keyed by report path; raw files are immutable) so the
    # build does not re-parse every injury report; copied, so the real cache is untouched
    p.interim.mkdir(parents=True)
    for f in PATHS.interim.glob("injury_reports_parsed*.parquet"):
        shutil.copy2(f, p.interim / f.name)
    return build_all(p, load_configs(PATHS)), p


def test_build_produces_validated_tables(built):
    tables, p = built
    g = tables["games"]
    assert g.game_id.is_unique
    per_season = g.groupby("season").size()
    # 1230 games except: 2011-12 lockout (990), 2012-13 cancelled BOS-IND (1229), COVID seasons
    short = {"2011-12": 990, "2012-13": 1229, "2019-20": 1059, "2020-21": 1080}
    assert per_season.drop(list(short)).eq(1230).all()
    assert all(per_season[s] == n for s, n in short.items())
    assert tables["playoff_games"].groupby("season").size().between(79, 90).all()
    f = tables["team_game_features"]
    assert len(f) == 2 * len(g)
    assert (f.as_of_timestamp.dropna() < f.tipoff_utc[f.as_of_timestamp.notna()]).all()
    assert (p.reports / "data_audit.md").exists()
    checks = pd.read_csv(p.reports / "validation_results.csv")
    assert not ((checks.severity == "error") & ~checks.passed).any()


def test_baseline_beats_constant_out_of_sample(built):
    bt = pd.read_csv(built[1].reports / "baseline_backtest.csv")
    pooled = bt[bt.test_season == "pooled"].iloc[0]
    assert pooled.preseason_log_loss < pooled.constant_log_loss
    assert pooled.preseason_brier < pooled.constant_brier
