"""Data validation rules. Each check returns CheckResult records; ``severity='error'``
results make the build fail (after the audit report has been written)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class CheckResult:
    check: str
    season: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" | "warning"


def _res(check, season, bad, detail_ok, detail_bad, severity="error"):
    return CheckResult(check, season, not bad, detail_ok if not bad else detail_bad, severity)


def check_games(games: pd.DataFrame, season: str, max_poss_diff: float = 5.0) -> list[CheckResult]:
    g = games
    out = []
    teams = pd.unique(pd.concat([g.home_team_id, g.away_team_id]).dropna())
    out.append(_res("team_count", season, len(teams) != 30, "30 teams", f"{len(teams)} teams"))
    dup = int(g.game_id.duplicated().sum())
    out.append(_res("game_id_unique", season, dup > 0, "unique", f"{dup} duplicate game_id"))
    unpaired = int((g._pair != "both").sum())
    out.append(_res("two_team_rows_per_game", season, unpaired > 0, "every game has home+away rows",
                    f"{unpaired} games missing a team row"))
    not_sched = int((g._in_schedule != "both").sum())
    out.append(_res("game_in_schedule", season, not_sched > 0, "every game found in schedule",
                    f"{not_sched} games not in schedule"))
    per_team = pd.concat([g.home_team_id, g.away_team_id]).value_counts()
    out.append(_res("equal_games_per_team", season, per_team.nunique() != 1,
                    f"every team plays {per_team.iloc[0]}",
                    f"games per team range {per_team.min()}-{per_team.max()}"))
    exp = len(teams) * per_team.max() / 2
    out.append(_res("game_count", season, len(g) != exp, f"{len(g)} games",
                    f"{len(g)} games, expected {exp:.0f}"))
    same = int((g.home_team_id == g.away_team_id).sum())
    out.append(_res("home_ne_away", season, same > 0, "ok", f"{same} games with home==away"))
    home_mis = int((g.home_team_id != g.sched_home_team_id).sum() + (g.away_team_id != g.sched_away_team_id).sum())
    out.append(_res("home_matches_schedule", season, home_mis > 0, "ok", f"{home_mis} mismatches"))
    mu = g[~g.is_neutral]
    mu_bad = int((~mu.home_matchup_says_home.fillna(False)).sum() + mu.away_matchup_says_home.fillna(True).sum())
    out.append(_res("matchup_string_home_flag", season, mu_bad > 0,
                    "non-neutral games: MATCHUP 'vs.' agrees with schedule home team",
                    f"{mu_bad} non-neutral team rows disagree"))
    null_scores = int(g.home_score.isna().sum() + g.away_score.isna().sum())
    out.append(_res("scores_present", season, null_scores > 0, "ok", f"{null_scores} null scores"))
    ties = int((g.home_score == g.away_score).sum())
    out.append(_res("no_ties", season, ties > 0, "ok", f"{ties} ties"))
    wl_bad = int(((g.home_wl == "W") != (g.home_score > g.away_score)).sum()
                 + ((g.away_wl == "W") != (g.away_score > g.home_score)).sum())
    out.append(_res("wl_matches_score", season, wl_bad > 0, "ok", f"{wl_bad} rows disagree"))
    pm_bad = int((g.home_plus_minus != g.margin).sum())
    out.append(_res("plus_minus_matches_margin", season, pm_bad > 0, "ok", f"{pm_bad} games disagree"))
    poss_na = int(g.possessions.isna().sum())
    out.append(_res("possessions_present", season, poss_na > 0, "ok", f"{poss_na} games without POSS"))
    pdiff = (g.home_poss_source - g.away_poss_source).abs()
    out.append(_res("possession_source_agreement", season, bool((pdiff > max_poss_diff).any()),
                    f"max |POSS_home-POSS_away| = {pdiff.max():.0f}; equal in {(pdiff == 0).mean():.1%} of games",
                    f"{int((pdiff > max_poss_diff).sum())} games differ by >{max_poss_diff}", "warning"))
    date_mis = int((g.game_date != g.sched_game_date).sum())
    out.append(_res("game_date_matches_schedule", season, date_mis > 0, "ok", f"{date_mis} mismatches"))
    tip_na = int(g.tipoff_utc.isna().sum())
    out.append(_res("tipoff_present", season, tip_na > 0, "ok", f"{tip_na} missing tipoff"))
    et_date = g.tipoff_utc.dt.tz_convert("America/New_York").dt.date
    et_bad = int((et_date != g.game_date).sum())
    out.append(_res("tipoff_et_date_matches_game_date", season, et_bad > 0, "ok",
                    f"{et_bad} games whose ET tip-off date differs from game_date"))
    v_na = int(g.venue_id.isna().sum())
    out.append(_res("venue_mapped", season, v_na > 0, "ok", f"{v_na} games without venue_id"))
    unflagged = g[g.is_neutral & ~g.source_is_neutral.fillna(False).astype(bool)]
    out.append(CheckResult("neutral_flag_source_vs_venue", season, True,
                           f"{int(g.is_neutral.sum())} neutral games; {len(unflagged)} neutral by venue "
                           f"but not flagged by source: {unflagged.game_id.tolist()}", "warning"))
    # one game per team per ET date -> 'strictly earlier date' == 'previous games'
    td = pd.concat([g[["home_team_id", "game_date"]].set_axis(["t", "d"], axis=1),
                    g[["away_team_id", "game_date"]].set_axis(["t", "d"], axis=1)])
    dd = int(td.duplicated().sum())
    out.append(_res("one_game_per_team_per_date", season, dd > 0, "ok", f"{dd} team-dates with 2 games"))
    return out


def check_schedule(schedule: pd.DataFrame, season: str) -> list[CheckResult]:
    s = schedule
    out = []
    na = int(s.conference.isna().sum() + s.division.isna().sum())
    out.append(_res("conference_division_present", season, na > 0, "ok", f"{na} missing"))
    t = s.drop_duplicates("team_id")
    conf = t.conference.value_counts()
    div = t.division.value_counts()
    bad = not (len(conf) == 2 and (conf == 15).all() and len(div) == 6 and (div == 5).all())
    out.append(_res("conference_division_sizes", season, bad, "2x15 conferences, 6x5 divisions",
                    f"conferences {conf.to_dict()} divisions {div.to_dict()}"))
    coord_na = int(s.arena_latitude.isna().sum())
    out.append(_res("venue_coordinates_present", season, coord_na > 0, "ok", f"{coord_na} rows"))
    return out


def check_local_time(games: pd.DataFrame, schedule: pd.DataFrame, season: str) -> list[CheckResult]:
    """Derived venue-local tip-off vs the source's home-team local time (non-neutral only).
    The source stores a wall-clock string with a misleading 'Z'; only HH:MM is compared."""
    home = schedule[(schedule.home_away == "H") & ~schedule.is_neutral][["game_id", "tipoff_local_time"]]
    m = home.merge(games[["game_id", "home_team_local_time"]], on="game_id")
    src = m.home_team_local_time.astype(str).str[11:16]
    mine = m.tipoff_local_time.astype(str).str[11:16]
    bad = m[src != mine]
    return [_res("venue_timezone_vs_source_local_time", season, len(bad) > 0,
                 f"{len(m)} home games agree",
                 f"{len(bad)} of {len(m)} disagree, e.g. {bad.game_id.head(5).tolist()}", "warning")]


def check_player_minutes(player_games: pd.DataFrame, team_minutes: pd.DataFrame, season: str,
                         tol: float = 1.0) -> list[CheckResult]:
    s = player_games.groupby(["game_id", "team_id"]).minutes.sum().rename("player_min").reset_index()
    m = team_minutes.merge(s, on=["game_id", "team_id"], how="left")
    missing = int(m.player_min.isna().sum())
    diff = (m.player_min - m.team_min).abs()
    bad = int((diff > tol).sum())
    return [_res("player_logs_cover_team_games", season, missing > 0, "ok",
                 f"{missing} team-games without player rows"),
            _res("player_minutes_sum_to_team_minutes", season, bad > 0,
                 f"max diff {diff.max():.2f} min", f"{bad} team-games differ by >{tol} min", "warning")]


def check_playoff_games(games: pd.DataFrame, season: str) -> list[CheckResult]:
    g, out = games, []
    out.append(_res("playoff_two_team_rows", season, bool((g._pair != "both").any()), "ok", "games missing a team row"))
    out.append(_res("playoff_game_in_schedule", season, bool((g._in_schedule != "both").any()), "ok", "games not in schedule"))
    out.append(_res("playoff_game_id_unique", season, bool(g.game_id.duplicated().any()), "ok", "duplicate game_id"))
    out.append(_res("playoff_venue_mapped", season, bool(g.venue_id.isna().any()), "ok", "games without venue"))
    out.append(_res("playoff_possessions_present", season, bool(g.possessions.isna().any()), "ok", "missing POSS"))
    wl = int(((g.home_wl == "W") != (g.home_score > g.away_score)).sum())
    out.append(_res("playoff_wl_matches_score", season, wl > 0, "ok", f"{wl} games disagree"))
    return out


def apply_known_exceptions(checks: pd.DataFrame, known: dict) -> pd.DataFrame:
    """Downgrade documented, expected failures to warnings and append the reason."""
    c = checks.copy()
    for season, rules in (known or {}).items():
        for check, reason in rules.items():
            m = (c.season == season) & (c.check == check) & ~c.passed
            c.loc[m, "severity"] = "warning"
            c.loc[m, "detail"] = c.loc[m, "detail"] + f" (known exception: {reason})"
    return c


def results_frame(results: list[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])
