"""Immutable raw-data store with an append-only manifest.

Every raw payload is written exactly as received, together with a manifest record
holding the source, request parameters, fetch time (UTC), HTTP metadata and sha256.
The upstream sources expose no per-record update timestamps, so ``fetched_at`` is the
only provenance time we can honestly record.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RawStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.jsonl"

    def path_for(self, relpath: str) -> Path:
        return self.root / relpath

    def exists(self, relpath: str) -> bool:
        return self.path_for(relpath).exists()

    def write(self, relpath: str, payload: bytes, meta: dict[str, Any]) -> dict[str, Any]:
        path = self.path_for(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        record = {
            "path": relpath,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "fetched_at": utc_now_iso(),
            **meta,
        }
        self.append_manifest(record)
        return record

    def append_manifest(self, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        with open(self.manifest_path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def latest_records(self) -> dict[str, dict[str, Any]]:
        """Latest manifest record per key (path, or url for not-found probes)."""
        out: dict[str, dict[str, Any]] = {}
        for rec in self.read_manifest():
            out[rec.get("path") or rec.get("url")] = rec
        return out
