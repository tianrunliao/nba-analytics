"""Actual meeting-count matrix N_ij and its comparison with the traditional 82-game
structure (which is only used as a reference, never as a rule)."""

from __future__ import annotations

import pandas as pd


def matchup_counts(schedule: pd.DataFrame) -> pd.DataFrame:
    """Long form: season, team_id, opponent_id, n_games, relation."""
    s = schedule
    n = s.groupby(["season", "team_id", "team", "opponent_id", "opponent"]).size().rename("n_games").reset_index()
    rel = s.drop_duplicates(["season", "team_id", "opponent_id"])[
        ["season", "team_id", "opponent_id", "conference", "division", "opponent_conference", "opponent_division"]
    ]
    n = n.merge(rel, on=["season", "team_id", "opponent_id"])
    n["relation"] = "other_conference"
    n.loc[n.conference == n.opponent_conference, "relation"] = "same_conference_other_division"
    n.loc[n.division == n.opponent_division, "relation"] = "same_division"
    return n.drop(columns=["conference", "division", "opponent_conference", "opponent_division"])


def matrix(counts: pd.DataFrame, season: str) -> pd.DataFrame:
    c = counts[counts.season == season]
    return c.pivot_table(index="team", columns="opponent", values="n_games", fill_value=0, aggfunc="sum")


# Traditional structure (plan section 2): relation -> {meetings: number of opponents}
TRADITIONAL = {
    "same_division": {4: 4},
    "same_conference_other_division": {4: 6, 3: 4},
    "other_conference": {2: 15},
}


def structure_deviations(counts: pd.DataFrame) -> pd.DataFrame:
    """Per team-season: opponents-by-meetings profile per relation vs the traditional one."""
    prof = counts.groupby(["season", "team", "relation", "n_games"]).size().rename("n_opponents").reset_index()
    rows = []
    for (season, team, relation), grp in prof.groupby(["season", "team", "relation"]):
        got = dict(zip(grp.n_games, grp.n_opponents))
        exp = TRADITIONAL[relation]
        if got != exp:
            rows.append(
                {
                    "season": season,
                    "team": team,
                    "relation": relation,
                    "actual": str(dict(sorted(got.items()))),
                    "traditional": str(exp),
                }
            )
    return pd.DataFrame(rows, columns=["season", "team", "relation", "actual", "traditional"])


def symmetry_violations(counts: pd.DataFrame) -> int:
    a = counts[["season", "team_id", "opponent_id", "n_games"]]
    b = a.rename(columns={"team_id": "opponent_id", "opponent_id": "team_id", "n_games": "n_rev"})
    m = a.merge(b, on=["season", "team_id", "opponent_id"], how="outer")
    return int((m.n_games != m.n_rev).sum())
