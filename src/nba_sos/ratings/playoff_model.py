"""Playoff "pure strength" ranking, trained on earlier postseasons only.

For a target season Y, every input is known before Y's first playoff game:

  theta    schedule/venue/rest/health-adjusted regular-season strength (team_strength),
           fit on regular seasons <= Y (Y's regular season is complete by then).
  momentum recency-weighted strength minus theta (tau chosen on training postseasons).
  h_start  health effect at playoff start, per 100 possessions: the regular-season
           coefficients applied to the players expected to miss games — from the official
           report before the team's first playoff game (P(miss | status) estimated on
           earlier regular seasons), or, where no usable report exists (seasons before the
           archive), from rotation players (RS mpg >= 20) who missed at least the team's last
           ``k`` regular-season games in a row. ``k`` is chosen by :func:`choose_streak_threshold`
           against the official pre-playoff Out/Doubtful status of report seasons; it uses
           no playoff result.

Playoff game model, fit on postseasons < Y:

    y = c1*d_theta + c2*d_momentum + c3*d_h_start + c4*d_h_inplay + HCA_po*[not neutral] + e

``d_h_inplay`` is the change in health during the playoffs (rotation players who were
available at the start but missed a playoff game). It is a control in training so
playoff injuries do not distort c1..c3, and it is set to 0 in the ranking:

    rating = c1*theta + c2*momentum + c3*h_start
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

ROTATION_MPG = 20.0
TAU_GRID = (10.0, 20.0, 40.0)


@dataclass
class SeasonInputs:
    season: str
    fit: dict
    rg: pd.DataFrame
    values: pd.DataFrame
    game_frame: pd.DataFrame


def status_miss_table(availability: pd.DataFrame, player_games: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """P(did not play | pre-game status, rotation player?) on regular-season reports of
    ``seasons``. Rotation = season minutes per game >= ROTATION_MPG: an 'Available' bench
    player often sits by coach's decision, an 'Available' starter almost never does."""
    a = availability[availability.season.isin(seasons) & availability.player_id.notna()]
    played = player_games[player_games.minutes > 0][["game_id", "player_id"]].assign(played=True)
    mpg = player_games[player_games.minutes > 0].groupby(["season", "player_id"]).minutes.mean().rename("mpg")
    m = a.merge(played, on=["game_id", "player_id"], how="left").merge(
        mpg.reset_index(), on=["season", "player_id"], how="left"
    )
    m["missed"] = ~m.played.astype("boolean").fillna(False).astype(bool)
    m["rotation"] = m.mpg.fillna(0) >= ROTATION_MPG
    return m.groupby(["injury_status", "rotation"]).missed.agg(p_miss="mean", n="size").reset_index()


def season_end_roster(rg: pd.DataFrame, values: pd.DataFrame, season: str) -> pd.DataFrame:
    r = rg[(rg.season == season) & rg.stint_to_season_end][["season", "player_id", "team_id"]].drop_duplicates()
    return r.merge(values, on=["season", "player_id", "team_id"])


def rs_end_absence_streak(rg: pd.DataFrame, season: str) -> pd.DataFrame:
    """player_id, team_id, streak: consecutive roster games the player missed at the end of
    ``season``'s regular season (season-end stint only). Known before the playoffs."""
    r = rg[(rg.season == season) & rg.stint_to_season_end].sort_values(["player_id", "team_id", "game_date"])
    rows = []
    for (pid, tid), g in r.groupby(["player_id", "team_id"], sort=False):
        played = g.played.to_numpy(bool)[::-1]
        rows.append((pid, tid, int(np.argmax(played)) if played.any() else len(played)))
    return pd.DataFrame(rows, columns=["player_id", "team_id", "streak"])


STREAK_GRID = range(1, 11)


def choose_streak_threshold(x: pd.DataFrame) -> dict:
    """``x``: rotation players of teams with a usable pre-playoff report, with ``streak`` and
    ``out`` (listed Out or Doubtful). Returns the k in STREAK_GRID whose rule streak >= k
    agrees best (F1) with the report, plus the scores of every k."""
    scores = {}
    for k in STREAK_GRID:
        flag = x.streak >= k
        tp = int((flag & x.out).sum())
        prec, rec = tp / max(int(flag.sum()), 1), tp / max(int(x.out.sum()), 1)
        scores[k] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    best = max(scores, key=lambda k: scores[k])
    return {"k": best, "f1": scores, "n_players": int(len(x)), "n_out": int(x.out.sum())}


def health_start(
    roster: pd.DataFrame,
    statuses: pd.DataFrame | None,
    rs_end_streak: pd.DataFrame | None,
    miss_table: pd.DataFrame | None,
    coefs: dict,
    streak_k: int | None = None,
) -> pd.DataFrame:
    """Per team: expected missing f and f*value at playoff start, and the per-100 effect.
    ``statuses``: player_id, team_id, injury_status from the pre-playoff report (or None).
    Without statuses: ``rs_end_streak`` (player_id, team_id, streak from
    :func:`rs_end_absence_streak`) and ``streak_k`` — a rotation player whose streak >= k
    is expected to miss."""
    r = roster.copy()
    r["f"] = r.mpg / 48
    if statuses is not None and len(statuses):
        assert miss_table is not None, "statuses need a miss_table"
        r = r.merge(statuses[["player_id", "team_id", "injury_status"]], on=["player_id", "team_id"], how="left")
        r["rotation"] = r.mpg >= ROTATION_MPG
        r = r.merge(miss_table[["injury_status", "rotation", "p_miss"]], on=["injury_status", "rotation"], how="left")
        r["p_miss"] = r.p_miss.fillna(0.0)  # not on the report -> available
        r["source"] = "injury_report"
    else:
        r = r.merge(rs_end_streak, on=["player_id", "team_id"], how="left")
        r["p_miss"] = ((r.streak.fillna(0) >= streak_k) & (r.mpg >= ROTATION_MPG)).astype(float)
        r["injury_status"] = np.where(r.p_miss > 0, "absent_rs_end", None)
        r["source"] = "rs_end_absence"
    r["exp_f"] = r.p_miss * r.f
    r["exp_fv"] = r.p_miss * r.f * r.value
    team = (
        r.groupby("team_id")
        .agg(h_minutes=("exp_f", "sum"), h_value=("exp_fv", "sum"), source=("source", "first"))
        .reset_index()
    )
    team["h_start"] = coefs["b_min"] * team.h_minutes + coefs["b_val"] * team.h_value
    return team, r


def inplay_health(
    playoff_pg: pd.DataFrame, roster: pd.DataFrame, start_detail: pd.DataFrame, playoff_games: pd.DataFrame, coefs: dict
) -> pd.DataFrame:
    """Per playoff team-game: health effect of rotation players who were expected to be
    available at the start (p_miss < 0.5) but logged no minutes in that game."""
    avail = start_detail[(start_detail.p_miss < 0.5) & (start_detail.mpg >= ROTATION_MPG)][
        ["player_id", "team_id", "f", "value"]
    ]
    tg = pd.concat(
        [
            playoff_games[["game_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
            playoff_games[["game_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
        ]
    )
    tg["team_id"] = tg.team_id.astype(int)
    x = tg.merge(avail.assign(team_id=avail.team_id.astype(int)), on="team_id")
    played = playoff_pg[playoff_pg.minutes > 0][["game_id", "player_id"]].assign(played=True)
    x = x.merge(played, on=["game_id", "player_id"], how="left")
    x = x[x.played.isna()]
    x["fv"] = x.f * x.value
    out = (
        x.groupby(["game_id", "team_id"])
        .agg(ip_minutes=("f", "sum"), ip_value=("fv", "sum"), ip_players=("player_id", "size"))
        .reset_index()
    )
    out["h_inplay"] = coefs["b_min"] * out.ip_minutes + coefs["b_val"] * out.ip_value
    return out, x


def team_table(theta: pd.DataFrame, recent: pd.DataFrame, hs: pd.DataFrame, season: str) -> pd.DataFrame:
    t = recent[["team_id", "theta_full", "momentum"]].merge(
        hs[["team_id", "h_start", "h_minutes", "h_value", "source"]], on="team_id", how="left"
    )
    t["h_start"] = t.h_start.fillna(0.0)
    t["season"] = season
    return t


TEAM_COLS = ["theta_full", "momentum", "h_start"]


def playoff_rows(
    pgames: pd.DataFrame, teams: pd.DataFrame, inplay: pd.DataFrame, team_cols: list[str] | None = None
) -> pd.DataFrame:
    cols = team_cols or TEAM_COLS
    d = pgames[
        ["season", "game_id", "home_team_id", "away_team_id", "is_neutral", "net_rating_margin", "home_win"]
    ].copy()
    d["home_team_id"], d["away_team_id"] = d.home_team_id.astype(int), d.away_team_id.astype(int)
    tt = teams.assign(team_id=teams.team_id.astype(int))
    for side in ("home", "away"):
        d = d.merge(tt[["team_id"] + cols].rename(columns=lambda c: f"{side}_{c}"), on=f"{side}_team_id", how="left")
        ip = inplay[["game_id", "team_id", "h_inplay"]].rename(
            columns={"team_id": f"{side}_team_id", "h_inplay": f"{side}_h_inplay"}
        )
        ip[f"{side}_team_id"] = ip[f"{side}_team_id"].astype(int)
        d = d.merge(ip, on=["game_id", f"{side}_team_id"], how="left")
        d[f"{side}_h_inplay"] = d[f"{side}_h_inplay"].fillna(0.0)
    for c in cols + ["h_inplay"]:
        d[f"d_{c}"] = d[f"home_{c}"] - d[f"away_{c}"]
    d["home_flag"] = (~d.is_neutral).astype(float)
    d["y"] = d.net_rating_margin.astype(float)
    return d


FEATS = ["d_theta_full", "d_momentum", "d_h_start", "d_h_inplay", "home_flag"]
NOT_TEAM = {"home_flag", "d_h_inplay"}


def fit_playoff(rows: pd.DataFrame, feats: list[str] | None = None) -> dict:
    feats = feats or FEATS
    X = rows[feats].to_numpy(float)
    y = rows.y.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma = float(np.sqrt(resid @ resid / dof))
    cov = sigma**2 * np.linalg.inv(X.T @ X)
    return {
        "coef": dict(zip(feats, beta)),
        "se": dict(zip(feats, np.sqrt(np.diag(cov)))),
        "sigma": sigma,
        "n_games": int(len(y)),
        "loglik": float(norm.logpdf(resid, scale=sigma).sum()),
    }


def rating(teams: pd.DataFrame, pm: dict) -> pd.Series:
    """Team rating: every team-level coefficient applied to the team table (no home court,
    no in-playoff health)."""
    out = 0.0
    for f, c in pm["coef"].items():
        if f not in NOT_TEAM:
            out = out + c * teams[f[2:]]
    return out


def predict_rows(rows: pd.DataFrame, pm: dict, with_inplay: bool = False) -> np.ndarray:
    m = 0.0
    for f, c in pm["coef"].items():
        if f == "d_h_inplay" and not with_inplay:
            continue
        m = m + c * rows[f]
    return np.asarray(m, float)


def series_win_prob(r_a: float, r_b: float, a_has_home: bool, hca: float, sigma: float) -> float:
    """P(team A wins a best-of-7, 2-2-1-1-1 home pattern), games independent."""
    pattern = [1, 1, 0, 0, 1, 0, 1] if a_has_home else [0, 0, 1, 1, 0, 1, 0]
    p = [norm.cdf((r_a - r_b + (hca if h else -hca)) / sigma) for h in pattern]
    # dynamic programming over (wins_a, wins_b)
    dp = {(0, 0): 1.0}
    win = 0.0
    for gi in range(7):
        nxt: dict[tuple[int, int], float] = {}
        for (wa, wb), pr in dp.items():
            for res, q in ((1, p[gi]), (0, 1 - p[gi])):
                na, nb = wa + res, wb + (1 - res)
                if na == 4:
                    win += pr * q
                elif nb < 4:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + pr * q
        dp = nxt
    return win
