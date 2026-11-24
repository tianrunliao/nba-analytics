"""Official NBA injury reports (PDF), fetched as pre-game snapshots.

For every regular-season game we try the report slot right before tip-off, stepping
back to earlier slots if a file does not exist. Each download keeps the HTTP
``Last-Modified`` header, used as the publication time. Selection of the report for a
game is done offline in :mod:`nba_sos.preprocessing.availability`, which requires
``published_at < tipoff``.

URL naming (see configs/data_sources.yaml): before ``new_start_et`` files are hourly,
``..._05PM.pdf`` being published at 17:30 ET; from then on they are quarter-hourly,
``..._05_30PM.pdf`` being published at 17:30 ET.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Callable
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .raw_store import RawStore

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

VALID_STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
# The CDN answers 403 (not 404) for files that do not exist, e.g. every slot before the
# archive starts in 2018-19; both are recorded once and not re-requested.
MISSING_STATUSES = {403, 404}
NETWORK_RETRIES = 3


# --------------------------------------------------------------------------- slots
def slot_nominal_times(tipoff_et: datetime, new_start_et: datetime, lookback: int) -> list[datetime]:
    """Candidate report times (ET, naive wall clock) strictly before tip-off, latest first."""
    t = tipoff_et.replace(tzinfo=None, second=0, microsecond=0)
    out = []
    if t < new_start_et:
        # hourly file H is published at H:30
        c = t.replace(minute=30) if t.minute > 30 else (t - timedelta(hours=1)).replace(minute=30)
        for _ in range(lookback):
            out.append(c)
            c -= timedelta(hours=1)
    else:
        c = t.replace(minute=(t.minute // 15) * 15)
        if c >= t:
            c -= timedelta(minutes=15)
        for _ in range(lookback):
            out.append(c)
            c -= timedelta(minutes=15)
    return out


def slot_filename(nominal_et: datetime, new_start_et: datetime) -> str:
    if nominal_et < new_start_et:
        stem = (nominal_et.replace(minute=0)).strftime("%I%p")
    else:
        stem = nominal_et.strftime("%I_%M%p")
    return f"Injury-Report_{nominal_et:%Y-%m-%d}_{stem}.pdf"


def raw_relpath(filename: str) -> str:
    return f"injury_reports/{filename[14:24]}/{filename}"


# --------------------------------------------------------------------------- fetch
def http_client(url: str, timeout: int, user_agent: str) -> tuple[int, bytes, str | None]:
    r = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    return r.status_code, (r.content if r.status_code == 200 else b""), r.headers.get("Last-Modified")


def fetch_report(
    store: RawStore, filename: str, cfg: dict, known: dict, client: Callable = http_client, sleep: Callable = time.sleep
) -> dict:
    """Download one report unless already known (downloaded or recorded as missing)."""
    url = cfg["base_url"] + filename[len("Injury-Report_") :]
    rel = raw_relpath(filename)
    if rel in known:
        return known[rel]
    if url in known and known[url].get("http_status") in MISSING_STATUSES:
        return known[url]
    for attempt in range(1, NETWORK_RETRIES + 1):
        try:
            status, body, last_modified = client(url, int(cfg["request_timeout_s"]), cfg["user_agent"])
            break
        except requests.exceptions.RequestException as err:  # DNS / connection / timeout
            if attempt == NETWORK_RETRIES:
                log.warning(
                    "injury report %s: network error after %d attempts (%s); will retry next run",
                    url,
                    attempt,
                    type(err).__name__,
                )
                return {"path": None, "url": url, "http_status": None, "fetched_at": None}
            sleep(10.0 * attempt)
    sleep(float(cfg["sleep_between_requests_s"]))
    published = parsedate_to_datetime(last_modified).astimezone(UTC).isoformat() if last_modified else None
    if status == 200 and body[:4] == b"%PDF":
        rec = store.write(
            rel,
            body,
            {
                "source": "official.nba.com injury report",
                "url": url,
                "http_status": 200,
                "last_modified": last_modified,
                "published_at_utc": published,
            },
        )
        known[rel] = rec
        return rec
    rec = {"path": None, "url": url, "http_status": status, "fetched_at": None}
    if status in MISSING_STATUSES:  # permanent absence; anything else is retried next run
        from .raw_store import utc_now_iso

        rec["fetched_at"] = utc_now_iso()
        store.append_manifest(rec)
        known[url] = rec
    else:
        log.warning("injury report %s -> HTTP %s", url, status)
    return rec


def fetch_for_games(
    store: RawStore, games: pd.DataFrame, cfg: dict, client: Callable = http_client, sleep: Callable = time.sleep
) -> pd.DataFrame:
    """``games`` needs ``game_id`` and ``tipoff_utc`` (tz-aware). For each distinct tip-off,
    walk back through slots until a report published before tip-off is found."""
    new_start = datetime.fromisoformat(cfg["new_start_et"])
    lookback = int(cfg["max_slot_lookback"])
    known = store.latest_records()
    resolved: dict[pd.Timestamp, str | None] = {}
    for tip in sorted(games.tipoff_utc.dropna().unique()):
        tip = pd.Timestamp(tip)
        tip_et = tip.tz_convert(ET).to_pydatetime()
        chosen = None
        for nominal in slot_nominal_times(tip_et, new_start, lookback):
            rec = fetch_report(store, slot_filename(nominal, new_start), cfg, known, client, sleep)
            pub = rec.get("published_at_utc")
            if rec.get("path") and pub and pd.Timestamp(pub) < tip:
                chosen = rec["path"]
                break
        resolved[tip] = chosen
    out = games[["game_id", "tipoff_utc"]].copy()
    out["report_path"] = out.tipoff_utc.map(resolved)
    return out


def fetch_for_seasons(paths, cfg: dict) -> None:
    """Regular-season snapshots for the configured seasons, plus the snapshot before
    each team's first playoff game for the configured playoff seasons."""
    from ..preprocessing.games import build_schedule_games
    from ..preprocessing.playoffs import first_playoff_games

    store = RawStore(paths.raw)
    icfg = cfg["data_sources"]["injury_reports"]
    for season in icfg["regular_season_seasons"]:
        res = fetch_for_games(store, build_schedule_games(store, season), icfg)
        log.info("%s RS: %d games, %d without a pre-tip-off report", season, len(res), res.report_path.isna().sum())
    for season in icfg["playoff_start_seasons"]:
        res = fetch_for_games(store, first_playoff_games(store, season), icfg)
        log.info("%s playoffs: %d first games, %d without a report", season, len(res), res.report_path.isna().sum())


# --------------------------------------------------------------------------- parse
# Report layouts: (normalised header keys, output column per header). 2018-19 has extra
# Category / Previous Status columns, part of 2019-20 a trailing Previous Status column;
# afterwards the 7-column layout.
LAYOUTS = [
    (
        ["gamedate", "gametime", "matchup", "team", "playername", "currentstatus", "reason"],
        ["game_date", "game_time", "matchup", "team", "player", "status", "reason"],
    ),
    (
        [
            "gamedate",
            "gametime",
            "matchup",
            "team",
            "playername",
            "category",
            "reason",
            "currentstatus",
            "previousstatus",
        ],
        ["game_date", "game_time", "matchup", "team", "player", "category", "reason", "status", "previous_status"],
    ),
    # part of 2019-20: 7-column order plus a trailing Previous Status column
    (
        ["gamedate", "gametime", "matchup", "team", "playername", "currentstatus", "reason", "previousstatus"],
        ["game_date", "game_time", "matchup", "team", "player", "status", "reason", "previous_status"],
    ),
    # Nov-Dec 2019: Reason before Current Status
    (
        ["gamedate", "gametime", "matchup", "team", "playername", "reason", "currentstatus", "previousstatus"],
        ["game_date", "game_time", "matchup", "team", "player", "reason", "status", "previous_status"],
    ),
]
COLS = ["game_date", "game_time", "matchup", "team", "player", "status", "reason"]
_NOT_SUBMITTED = "notyetsubmitted"


def norm_key(text: str) -> str:
    """Accent-free, lowercase, letters and digits only."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", t.lower())


def player_key_from_report(name: str) -> str:
    """'Porter Jr., Michael' / 'PorterJr.,Michael' -> 'michaelporterjr'."""
    if "," in name:
        last, first = name.split(",", 1)
        return norm_key(first + last)
    return norm_key(name)


def _column_starts(words: list[dict]) -> tuple[list[float], list[str], str] | None:
    """x positions of the header columns, the layout's column names and the joined header
    key, from the header line (spaced or unspaced text)."""
    lines: dict[int, list[dict]] = {}
    for w in words:
        lines.setdefault(round(w["top"]), []).append(w)
    for top in sorted(lines):
        ws = sorted(lines[top], key=lambda w: w["x0"])
        joined = norm_key("".join(w["text"] for w in ws))
        for keys, cols in LAYOUTS:
            if joined != "".join(keys):
                continue
            starts, acc, i = [], "", 0
            for w in ws:
                if acc == "":
                    starts.append(w["x0"])
                acc += norm_key(w["text"])
                if acc == keys[i]:
                    acc, i = "", i + 1
                    if i == len(keys):
                        return starts, cols, joined
            return None
    return None


def _group_lines(words: list[dict], tol: float = 2.0) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


_REPORT_TIME = re.compile(r"InjuryReport:(\d{2}/\d{2}/\d{2})(\d{2}:\d{2})(AM|PM)")


def parse_report_pdf(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return (rows, info). One row per listed player, plus one row per team marked
    NOT YET SUBMITTED (player is None). ``info`` carries the nominal report time and
    counts of lines that could not be placed."""
    import pdfplumber

    rows: list[dict] = []
    info: dict[str, Any] = {"report_time_et": None, "orphan_fragments": 0, "pages": 0, "header_found": False}
    state: dict[str, Any] = {c: None for c in ("game_date", "game_time", "matchup", "team")}
    starts: list[float] | None = None
    layout_cols: list[str] = []
    header_key: str | None = None
    with pdfplumber.open(path) as pdf:
        info["pages"] = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages):
            words = page.extract_words()
            found = _column_starts(words)
            if found:
                (starts, layout_cols, header_key), info["header_found"] = found, True
            if starts is None:
                raise ValueError(f"{path.name}: column header not found on page {page_no + 1}")
            bounds = [s - 4 for s in starts]
            page_rows: list[dict] = []
            fragments: list[tuple[float, str]] = []
            for line in _group_lines(words):
                text = norm_key("".join(w["text"] for w in line))
                if text.startswith("injuryreport"):
                    m = _REPORT_TIME.search("".join(w["text"] for w in line))
                    if m and info["report_time_et"] is None:
                        info["report_time_et"] = datetime.strptime(
                            f"{m.group(1)} {m.group(2)}{m.group(3)}", "%m/%d/%y %I:%M%p"
                        )
                    continue
                if text.startswith("page") or text == header_key:
                    continue
                cells: dict[str, Any] = {c: [] for c in layout_cols}
                for w in line:
                    idx = max(i for i, b in enumerate(bounds) if w["x0"] >= b) if w["x0"] >= bounds[0] else 0
                    cells[layout_cols[idx]].append(w["text"])
                cells = {c: " ".join(v) if v else None for c, v in cells.items()}
                if cells.get("category") and not cells.get("reason") and norm_key(cells["category"]) == _NOT_SUBMITTED:
                    cells["reason"] = cells["category"]
                top = line[0]["top"]
                for c in ("game_date", "game_time", "matchup", "team"):
                    if cells[c]:
                        state[c] = cells[c]
                if cells["player"] and cells["status"]:
                    page_rows.append(
                        {
                            **state,
                            "player": cells["player"],
                            "status": cells["status"],
                            "reason": cells["reason"] or "",
                            "top": top,
                            "page": page_no + 1,
                            "not_submitted": False,
                        }
                    )
                elif cells["reason"] and norm_key(cells["reason"]) == _NOT_SUBMITTED:
                    page_rows.append(
                        {
                            **state,
                            "player": None,
                            "status": None,
                            "reason": cells["reason"],
                            "top": top,
                            "page": page_no + 1,
                            "not_submitted": True,
                        }
                    )
                elif cells["reason"] or cells["player"]:
                    fragments.append((top, cells["reason"] or cells["player"]))
            # attach wrapped reason fragments to the vertically nearest player row
            players = [r for r in page_rows if not r["not_submitted"]]
            for top, frag in fragments:
                if not players:
                    info["orphan_fragments"] += 1
                    continue
                near = min(players, key=lambda r: (abs(r["top"] - top), r["top"] > top))
                near["reason"] = (
                    (near["reason"] + " " + frag).strip()
                    if near["top"] <= top
                    else (frag + " " + near["reason"]).strip()
                )
            rows.extend(page_rows)
    df = pd.DataFrame(rows, columns=COLS + ["top", "page", "not_submitted"])
    return df.drop(columns=["top"]), info


def resolve_rows(df: pd.DataFrame, nickname_by_tricode: dict[str, str]) -> pd.DataFrame:
    """Normalise parsed rows: dates, matchup tricodes, team tricode, player key."""
    out = df.copy()
    out["game_date"] = pd.to_datetime(out.game_date, format="%m/%d/%Y", errors="coerce").dt.date
    mt = out.matchup.fillna("").str.replace(" ", "").str.extract(r"^([A-Z]{3})@([A-Z]{3})$")
    out["away_tricode"], out["home_tricode"] = mt[0], mt[1]

    def team_of(row):
        # among the two matchup teams, the longest nickname that suffixes the team cell
        # ("charlottehornets" ends with both "hornets" and "nets")
        key = norm_key(row.team or "")
        hits = [
            t
            for t in (row.away_tricode, row.home_tricode)
            if isinstance(t, str) and t in nickname_by_tricode and key.endswith(nickname_by_tricode[t])
        ]
        if not hits:
            return None
        hits.sort(key=lambda t: len(nickname_by_tricode[t]), reverse=True)
        if len(hits) > 1 and len(nickname_by_tricode[hits[0]]) == len(nickname_by_tricode[hits[1]]):
            return None
        return hits[0]

    out["team_tricode"] = out.apply(team_of, axis=1)
    out["player_key"] = out.player.map(lambda p: player_key_from_report(p) if isinstance(p, str) else None)
    out["status"] = out.status.str.replace(" ", "")
    out["status_valid"] = out.status.isin(VALID_STATUSES) | out.not_submitted
    return out
