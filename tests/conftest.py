"""Shared test helpers.

``make_games`` builds a tiny synthetic league (hand-written rows) used ONLY to check
code logic against hand-computed answers. It is never mixed with real data.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
# Real NBA payloads (stats.nba.com JSON subset, official injury-report PDFs) are kept out of
# the public repository (NBA terms restrict redistribution); tests that parse them skip
# when they are absent. Regenerate with tests/fixtures/make_raw_subset.py and the URLs in
# tests/fixtures/README.md.
needs_raw_subset = pytest.mark.skipif(not (FIXTURES / "raw_subset" / "nba_stats").exists(),
                                      reason="tests/fixtures/raw_subset not present (not redistributed)")
needs_injury_pdfs = pytest.mark.skipif(not any((FIXTURES / "injury_reports").glob("*.pdf")),
                                       reason="tests/fixtures/injury_reports not present (not redistributed)")

TEAMS = {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}

# Four venues with known coordinates / zones (values chosen for hand-computable tests).
VENUES = pd.DataFrame({
    "venue_id": ["V_AAA", "V_BBB", "V_CCC", "V_DDD", "V_NEUTRAL"],
    "latitude": [0.0, 0.0, 40.7505, 34.0430, 48.8391],
    "longitude": [0.0, 1.0, -73.9935, -118.2672, 2.3795],
    "timezone": ["UTC", "UTC", "America/New_York", "America/Los_Angeles", "Europe/Paris"],
    "neutral_site": [False, False, False, False, True],
})


def make_games(rows: list[tuple]) -> pd.DataFrame:
    """rows: (season, 'YYYY-MM-DD', home, away, home_pts, away_pts[, possessions[, neutral_venue]])"""
    recs = []
    for i, r in enumerate(rows):
        season, d, h, a, hp, ap = r[:6]
        poss = r[6] if len(r) > 6 else 100.0
        neutral = bool(r[7]) if len(r) > 7 else False
        gd = date.fromisoformat(d)
        recs.append({
            "game_id": f"00{season[2:4]}{i:05d}", "season": season, "game_date": gd,
            "sched_game_date": gd,
            "tipoff_utc": pd.Timestamp(f"{d} 23:00", tz="UTC"),
            "home_team_id": TEAMS[h], "away_team_id": TEAMS[a],
            "sched_home_team_id": TEAMS[h], "sched_away_team_id": TEAMS[a],
            "home_team": h, "away_team": a, "home_score": hp, "away_score": ap,
            "home_wl": "W" if hp > ap else "L", "away_wl": "W" if ap > hp else "L",
            "home_plus_minus": hp - ap, "overtime_periods": 0, "home_poss_source": poss, "away_poss_source": poss,
            "home_matchup_says_home": not neutral, "away_matchup_says_home": False,
            "possessions": poss, "venue_id": "V_NEUTRAL" if neutral else f"V_{h}",
            "is_neutral": neutral, "source_is_neutral": neutral, "home_team_local_time": None,
            "_pair": "both", "_in_schedule": "both",
        })
    g = pd.DataFrame(recs)
    for c in ["home_team_id", "away_team_id", "sched_home_team_id", "sched_away_team_id",
              "home_score", "away_score", "overtime_periods"]:
        g[c] = g[c].astype("Int64")
    g["home_win"] = (g.home_score > g.away_score).astype("Int64")
    g["margin"] = (g.home_score - g.away_score).astype("Int64")
    g["home_off_rating"] = g.home_score / g.possessions * 100
    g["away_off_rating"] = g.away_score / g.possessions * 100
    g["net_rating_margin"] = g.home_off_rating - g.away_off_rating
    return g


@pytest.fixture
def venues() -> pd.DataFrame:
    return VENUES.copy()


@pytest.fixture
def mini_season() -> pd.DataFrame:
    """One season, 4 teams, 8 games over 6 dates."""
    return make_games([
        ("2022-23", "2022-10-20", "AAA", "BBB", 110, 100),
        ("2022-23", "2022-10-20", "CCC", "DDD", 95, 105),
        ("2022-23", "2022-10-21", "BBB", "CCC", 120, 90),   # BBB back-to-back (played 10-20)
        ("2022-23", "2022-10-23", "DDD", "AAA", 101, 99),
        ("2022-23", "2022-10-24", "AAA", "CCC", 100, 102),
        ("2022-23", "2022-10-24", "BBB", "DDD", 98, 97),
        ("2022-23", "2022-10-26", "CCC", "AAA", 88, 111),
        ("2022-23", "2022-10-27", "DDD", "BBB", 104, 100, 100.0, True),  # neutral site
    ])


EMPTY_AVAIL = pd.DataFrame(columns=["season", "game_id", "team_id", "team", "player_id", "player_name_raw",
                                    "player_key", "injury_status", "status_valid", "reason", "report_path",
                                    "report_timestamp", "match_method", "source"])


def team_status_all_unusable(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in games.itertuples():
        for tid in (r.home_team_id, r.away_team_id):
            rows.append({"season": r.season, "game_id": r.game_id, "team_id": tid, "report_path": None,
                         "report_published_at_utc": pd.NaT, "report_found": False, "game_in_report": False,
                         "not_submitted": False, "n_rows": 0, "snapshot_usable": False})
    ts = pd.DataFrame(rows)
    ts["team_id"] = ts.team_id.astype("Int64")
    ts["report_published_at_utc"] = pd.to_datetime(ts.report_published_at_utc, utc=True)
    return ts


EMPTY_PLAYER_GAMES = pd.DataFrame({"season": pd.Series(dtype=str), "game_id": pd.Series(dtype=str),
                                   "game_date": pd.Series(dtype=object), "player_id": pd.Series(dtype="Int64"),
                                   "player_name": pd.Series(dtype=str), "player_key": pd.Series(dtype=str),
                                   "team_id": pd.Series(dtype="Int64"), "minutes": pd.Series(dtype=float)})

FEATURE_CFG = {"schedule_density": {"windows_days": [4, 7]},
               "injury": {"missing_statuses": ["Out", "Doubtful"]}}
