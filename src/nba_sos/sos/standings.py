"""Official regular-season tie-break procedures (nba.com/standings, "Play-In and Playoff
Tie-Break Procedures", paragraph L.2; read 2026-11-23) for one simulated season.

Two teams (L.2.a): head-to-head win% | division winner | division record (same division
only) | conference record | record vs postseason-eligible teams of own conference | vs
postseason-eligible teams of the other conference | point differential.
More than two (L.2.b): division winner | win% in games among the tied teams | division
record (all in one division only) | conference record | vs postseason-eligible teams of own
conference | point differential.
Guidelines (L.2.c): division-winner ties are broken first; a criterion that separates some
but not all teams places the better group(s) higher and the teams still tied restart from
the first criterion of the list for their size; if nothing separates them, a random draw.
"Postseason-eligible": teams with one of the 10 best records of their conference, ties
for 10th included.
"""

from __future__ import annotations

import numpy as np


class Season:
    """One simulated (or real) season: wins, head-to-head wins W[i, j] (i beat j),
    point differential, conference and division of every team (array positions)."""

    def __init__(self, wins, W, pdiff, conf, div, rng):
        self.wins, self.W, self.pdiff = np.asarray(wins, float), np.asarray(W, float), np.asarray(pdiff, float)
        self.conf, self.div, self.rng = np.asarray(conf), np.asarray(div), rng
        self.div_winners: set[int] = set()
        self.eligible: dict = {}

    # -------------------------------------------------- helpers
    def pct(self, t: int, opp) -> float | None:
        opp = [o for o in opp if o != t]
        w = self.W[t, opp].sum()
        n = w + self.W[opp, t].sum()
        return None if n == 0 else round(float(w / n), 12)

    def _eligible(self, cname) -> list[int]:
        if cname not in self.eligible:
            cols = np.flatnonzero(self.conf == cname)
            tenth = np.sort(self.wins[cols])[::-1][min(9, len(cols) - 1)]
            self.eligible[cname] = list(cols[self.wins[cols] >= tenth])
        return self.eligible[cname]

    # -------------------------------------------------- criteria (None = not applicable)
    def _criteria(self, n: int, for_division: bool):
        same_div = lambda g: len({self.div[t] for t in g}) == 1  # noqa: E731
        h2h = lambda t, g: self.pct(t, g)  # noqa: E731
        divw = (lambda t, g: None) if for_division else (lambda t, g: float(t in self.div_winners))
        divrec = lambda t, g: self.pct(t, np.flatnonzero(self.div == self.div[t])) if same_div(g) else None  # noqa: E731
        confrec = lambda t, g: self.pct(t, np.flatnonzero(self.conf == self.conf[t]))  # noqa: E731
        elig_own = lambda t, g: self.pct(t, self._eligible(self.conf[t]))  # noqa: E731
        others = lambda c: [x for x in np.unique(self.conf) if x != c]  # noqa: E731
        elig_opp = lambda t, g: (  # noqa: E731
            self.pct(t, self._eligible(others(self.conf[t])[0])) if others(self.conf[t]) else None
        )
        pdiff = lambda t, g: round(float(self.pdiff[t]), 9)  # noqa: E731
        if n == 2:
            return [h2h, divw, divrec, confrec, elig_own, elig_opp, pdiff]
        return [divw, h2h, divrec, confrec, elig_own, pdiff]

    def break_ties(self, group: list[int], for_division: bool = False) -> list[int]:
        if len(group) <= 1:
            return list(group)
        for crit in self._criteria(len(group), for_division):
            vals = [crit(t, group) for t in group]
            if any(v is None for v in vals):
                continue
            levels = sorted(set(vals), reverse=True)
            if len(levels) == 1:
                continue
            out = []
            for lv in levels:  # better group(s) first; restart within
                out += self.break_ties([t for t, v in zip(group, vals) if v == lv], for_division)
            return out
        g = list(group)
        self.rng.shuffle(g)
        return g

    # -------------------------------------------------- standings
    def _by_wins(self, cols, for_division=False) -> list[int]:
        out = []
        for w in sorted(set(self.wins[cols]), reverse=True):
            out += self.break_ties([int(t) for t in cols if self.wins[t] == w], for_division)
        return out

    def conference_order(self, cname) -> list[int]:
        cols = np.flatnonzero(self.conf == cname)
        for d in np.unique(self.div[cols]):
            dc = cols[self.div[cols] == d]
            self.div_winners.add(self._by_wins(dc, for_division=True)[0])
        return self._by_wins(cols)
