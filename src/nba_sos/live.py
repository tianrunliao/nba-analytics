"""On-demand live projection: a fresh current-season snapshot, the latest official injury
report, and the remaining-season projection of the ``project`` command.

    python -m nba_sos.pipeline live     # network: stats.nba.com and official.nba.com

Each call fetches once; nothing is scheduled or refreshed automatically. The output goes to
``reports/live/<snapshot UTC stamp>/``: ``projection.csv`` (one row per team),
``injury_report.csv`` (the parsed official report, when one was found) and ``meta.json``.

The projection's availability terms come from game participation before the cutoff
(``sos.availability.state_at``), as in ``project``; in addition, for every team whose next
game the official report covers, that game uses P(miss | report status) from earlier
seasons' reports (``sos.availability.report_overlay``; checked by ``report-check``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd

from .config import Paths
from .ingestion import injury_reports as ir
from .ingestion.raw_store import RawStore

log = logging.getLogger(__name__)
# a slot is requested only once it is this old: the CDN answers 403 for a file that is not
# uploaded yet, and 403 is recorded as permanent absence (injury_reports.fetch_report)
MIN_SLOT_AGE = timedelta(minutes=30)
# walk back at most one day of quarter-hour slots (no report is published overnight)
LOOKBACK_SLOTS = 96


def latest_injury_report(
    paths: Paths,
    cfg: dict,
    now_utc: datetime | None = None,
    client: Callable = ir.http_client,
    sleep: Callable = time.sleep,
) -> dict | None:
    """Manifest record of the latest official report at least MIN_SLOT_AGE old, fetching
    slots from the newest back (already-known files are not requested again); None when no
    report exists in the last LOOKBACK_SLOTS slots."""
    ir_cfg = cfg["data_sources"]["injury_reports"]
    new_start = datetime.fromisoformat(ir_cfg["new_start_et"])
    store = RawStore(paths.raw)
    known = store.latest_records()
    start_et = ((now_utc or datetime.now(UTC)) - MIN_SLOT_AGE).astimezone(ir.ET)
    for nominal in ir.slot_nominal_times(start_et, new_start, LOOKBACK_SLOTS):
        rec = ir.fetch_report(store, ir.slot_filename(nominal, new_start), ir_cfg, known, client, sleep)
        if rec.get("path"):
            return rec
    return None


def parse_report(paths: Paths, rec: dict) -> pd.DataFrame:
    from .preprocessing.availability import nickname_keys

    df, _ = ir.parse_report_pdf(RawStore(paths.raw).path_for(rec["path"]))
    return ir.resolve_rows(df, nickname_keys())


def run_live(paths: Paths, cfg: dict, n_paths: int = 10000) -> dict:
    """Snapshot, latest report, projection; writes and returns the outputs."""
    from .ingestion.fetch_all import snapshot_current_season
    from .models import season_simulation as SIM

    snap = snapshot_current_season(paths, cfg)
    rec = latest_injury_report(paths, cfg)
    report = parse_report(paths, rec) if rec else None
    proj = SIM.run_current(paths, cfg, n_paths=n_paths, report=report, report_rec=rec)
    if proj["meta"]["snapshot"] != snap.name:
        raise RuntimeError(f"projection used snapshot {proj['meta']['snapshot']}, expected {snap.name}")
    dest = paths.reports / "live" / snap.name
    dest.mkdir(parents=True, exist_ok=True)
    proj["table"].to_csv(dest / "projection.csv", index=False)
    if report is not None:
        report.to_csv(dest / "injury_report.csv", index=False)
    meta = {
        **proj["meta"],
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "injury_report": None
        if rec is None
        else {k: rec.get(k) for k in ("path", "url", "published_at_utc", "sha256")},
        "injury_report_used_in_projection": proj["meta"]["injury_report_in_projection"] is not None,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("live outputs in %s", dest)
    return {"dir": dest, "table": proj["table"], "injury_report": report, "meta": meta}
