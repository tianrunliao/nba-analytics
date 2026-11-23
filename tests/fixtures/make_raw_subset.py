"""Regenerate tests/fixtures/raw_subset/ from a full raw store.

The fixtures are real stats.nba.com payloads trimmed to a handful of games — not
synthetic data. Run from the project root after `python -m nba_sos.pipeline fetch`:

    .venv/bin/python tests/fixtures/make_raw_subset.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "nba_stats"
DST = Path(__file__).resolve().parent / "raw_subset" / "nba_stats"

KEEP = {
    # 2023-24: Mexico City (neutral, not flagged by source), Las Vegas Cup semifinal,
    # two ordinary games, the Cup final (006) and one preseason game (001).
    "2023-24": ["0022300172", "0022301229", "0022300001", "0022300002", "0062300001", "0012300001"],
    # 2024-25: source-flagged neutral games whose MATCHUP strings use '@' for both teams.
    "2024-25": ["0022400147", "0022401229", "0022400001"],
}


def trim_result_sets(payload: dict, ids: set[str]) -> dict:
    for rs in payload["resultSets"]:
        i = rs["headers"].index("GAME_ID")
        rs["rowSet"] = [r for r in rs["rowSet"] if str(r[i]).zfill(10) in ids]
    return payload


def trim_schedule(payload: dict, ids: set[str]) -> dict:
    dates = []
    for gd in payload["leagueSchedule"]["gameDates"]:
        games = [g for g in gd["games"] if g["gameId"] in ids]
        if games:
            dates.append({**gd, "games": games})
    payload["leagueSchedule"]["gameDates"] = dates
    return payload


def main() -> None:
    for season, ids in KEEP.items():
        ids = set(ids)
        for name in ["league_game_log", "team_advanced"]:
            p = json.loads((SRC / name / f"{season}.json").read_text())
            out = DST / name / f"{season}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(trim_result_sets(p, ids)))
        p = json.loads((SRC / "schedule" / f"{season}.json").read_text())
        out = DST / "schedule" / f"{season}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(trim_schedule(p, ids)))
        st = DST / "standings" / f"{season}.json"
        st.parent.mkdir(parents=True, exist_ok=True)
        st.write_text((SRC / "standings" / f"{season}.json").read_text())


if __name__ == "__main__":
    main()
