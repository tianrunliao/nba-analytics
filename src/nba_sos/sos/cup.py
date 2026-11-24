"""NBA Cup (Emirates NBA Cup, from 2023-24) inside the season simulation.

Format (https://www.nba.com/news/nba-cup-101): six groups of five (three per
conference), four group games per team (regular-season games already on the schedule).
Per conference the three group winners and one wild card (best record among the
second-placed teams) reach the quarterfinals: seeds 1-3 = group winners by group record,
seed 4 = wild card; 1 hosts 4, 2 hosts 3. Semifinals: higher seed's arena (2026-27) or
neutral (Las Vegas, 2023-24..2025-26). Quarterfinals and semifinals count in the standings,
the final does not. The 22 non-qualifiers play two extra regular-season games, the two
quarterfinal losers of a conference play each other once, so everyone reaches 82.

Group tie-break order: head-to-head in group play, group point differential, total
points, previous regular-season record, random draw (ties restart within the still-tied
teams). Simplifications, stated: total points are not simulated (the criterion is
skipped); overtime is not separated from point differential; the official "formulaic"
pairing of the extra games (by group-play rank, two of them cross-conference) is not
published in detail, so extra games are paired at random within the conference, with no
pair repeated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CupPlan:
    groups: dict[str, list[int]]  # "East Group A" -> team ids
    group_games: pd.DataFrame  # game_id, home_team_id, away_team_id, group
    qf_date: pd.Timestamp
    sf_date: pd.Timestamp
    extra_dates: list[pd.Timestamp]
    sf_home_court: bool
    prev_wins: pd.Series  # team_id -> previous regular-season wins
    conference: dict[str, str] = field(default_factory=dict)  # group -> "East"/"West"


def plan(sched: pd.DataFrame, season: str, cup_cfg: dict, prev_wins: pd.Series) -> CupPlan | None:
    """``sched``: the season's schedule rows (game_id, date, home/away ids, game_sublabel,
    game_subtype, is_neutral). None for seasons without a Cup."""
    g = sched[sched.game_subtype.astype(str) == "in-season"]
    if g.empty:
        return None
    groups = {}
    for name, x in g.groupby("game_sublabel"):
        groups[name] = sorted(set(x.home_team_id.astype(int)) | set(x.away_team_id.astype(int)))
    ko = sched[sched.game_subtype.astype(str) == "in-season-knockout"]
    qf = ko[ko.game_sublabel.str.contains("Quarterfinal")]
    sf = ko[ko.game_sublabel.str.contains("Semifinal")]
    c = cup_cfg.get(season)
    if c is not None:
        extra = [pd.Timestamp(d) for d in c["extra_game_dates"]]
        sf_home = bool(c["semifinal_home_court"])
    else:
        ids = sched.game_id.astype(str).str[-5:].astype(int)
        mk = sched[(ids > 1200) & (sched.game_subtype.astype(str) != "in-season-knockout")]
        extra = [mk.date.min(), mk.date.max()]
        sf_home = not bool(sf.is_neutral.astype(bool).all())
    gg = g[["game_id", "home_team_id", "away_team_id", "game_sublabel"]].rename(columns={"game_sublabel": "group"})
    return CupPlan(
        groups=groups,
        group_games=gg.reset_index(drop=True),
        qf_date=qf.date.min(),
        sf_date=sf.date.min(),
        extra_dates=extra,
        sf_home_court=sf_home,
        prev_wins=prev_wins,
        conference={k: k.split()[0] for k in groups},
    )


def _break(group: list[int], crits, rng) -> list[int]:
    if len(group) <= 1:
        return list(group)
    for crit in crits:
        vals = [crit(t, group) for t in group]
        levels = sorted(set(vals), reverse=True)
        if len(levels) > 1:
            out = []
            for lv in levels:
                out += _break([t for t, v in zip(group, vals) if v == lv], crits, rng)
            return out
    g = list(group)
    rng.shuffle(g)
    return g


def group_tables(plan_: CupPlan, home_win: np.ndarray, margin: np.ndarray, rng) -> dict:
    """One path: per group the ordered teams, wins and point differential.
    home_win / margin (points, home view) are aligned with ``plan_.group_games``."""
    gg = plan_.group_games
    out = {}
    for name, teams in plan_.groups.items():
        sel = np.flatnonzero(gg.group.to_numpy() == name)
        w, pdiff = {t: 0 for t in teams}, {t: 0.0 for t in teams}
        beat: dict[tuple[int, int], int] = {}
        for i in sel:
            h, a = int(gg.home_team_id.iat[i]), int(gg.away_team_id.iat[i])
            win, loss = (h, a) if home_win[i] else (a, h)
            w[win] += 1
            beat[(win, loss)] = beat.get((win, loss), 0) + 1
            pdiff[h] += margin[i]
            pdiff[a] -= margin[i]

        def h2h(t, grp):
            ws = sum(beat.get((t, o), 0) for o in grp if o != t)
            ls = sum(beat.get((o, t), 0) for o in grp if o != t)
            return round(ws / (ws + ls), 12) if ws + ls else 0.5

        crits = [h2h, lambda t, grp: round(pdiff[t], 9), lambda t, grp: float(plan_.prev_wins.get(t, 0))]
        order = []
        for lv in sorted(set(w.values()), reverse=True):
            order += _break([t for t in teams if w[t] == lv], crits, rng)
        out[name] = {"order": order, "wins": w, "pdiff": pdiff}
    return out


def quarterfinal_seeds(plan_: CupPlan, tables: dict, rng) -> dict[str, list[int]]:
    """Conference -> [seed1, seed2, seed3, seed4(wild card)]."""
    crits = None
    out = {}
    for conf in ("East", "West"):
        names = [n for n in tables if plan_.conference[n] == conf]
        allw = {t: tables[n]["wins"][t] for n in names for t in tables[n]["wins"]}
        allp = {t: tables[n]["pdiff"][t] for n in names for t in tables[n]["pdiff"]}
        crits = [lambda t, g: round(allp[t], 9), lambda t, g: float(plan_.prev_wins.get(t, 0))]

        def rank(teams):
            o = []
            for lv in sorted({allw[t] for t in teams}, reverse=True):
                o += _break([t for t in teams if allw[t] == lv], crits, rng)
            return o

        winners = rank([tables[n]["order"][0] for n in names])
        wild = rank([tables[n]["order"][1] for n in names])[0]
        out[conf] = winners + [wild]
    return out


def pair_extra(teams: list[int], games_each: int, rng, tries: int = 200) -> list[tuple[int, int]]:
    """Random pairing of ``games_each`` slots per team, no self-pairs, no repeated pair."""
    slots = np.repeat(np.asarray(teams), games_each)
    for _ in range(tries):
        perm = rng.permutation(slots)
        a, b = perm[0::2], perm[1::2]
        pairs = {tuple(sorted(p)) for p in zip(a, b)}
        if not np.any(a == b) and len(pairs) == len(a):
            return list(zip(a.tolist(), b.tolist()))
    raise RuntimeError("could not pair the extra games")
