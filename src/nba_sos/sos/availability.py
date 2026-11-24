"""Future player availability for the remaining-season projection (plan 7.1 / 10.3).

State at a cutoff, from actual participation only (box scores, all seasons; the same
player-game frame as models/availability.py): every rotation player on a team's roster (his most recent
roster game before the cutoff was for that team; >= 20 mpg, >= 3 games played) with
  V  previous-season player value (LPM, minutes-shrunk; 0 without one)
  f  expected minutes / 48 (last-10 average, else season to date, else last season)
  n  consecutive team games missed going into the cutoff (0 = played the last one)

Missing value of a team-game: M_v = sum f V and M_f = sum f over players who do not play
(V is relative to an average player; the f term is the drop to replacement level, as in
models/availability.py). The cutoff strength already reflects what was missing in recent games, so the
forecast uses the change

    inj_v(k) = rec_v - E[M_v at the team's k-th next game],   same for f

rec = time-decayed mean (tau of the dynamic strength) of the actually missing M over the
team's games before the cutoff. Expectations:
  players out at the cutoff     P(still out at game k) = prod_{i<k} (1 - h(n + i)),
                                h(n) = P(plays next | missed n in a row) from earlier seasons
  players healthy at the cutoff pi(k) = P(misses game k | played the last game), earlier seasons
The Monte Carlo samples each injured player's return game from the same hazards; new
injuries enter through their expectation pi(k) (their randomness is part of the strength
random walk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths
from ..models import availability as AM

K_MAX = 90
N_MAX = 200


def player_frame(paths: Paths) -> pd.DataFrame:
    """Player-game frame of models/availability.py (cached in data/interim)."""
    cache = paths.interim / "player_game_frame.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = lambda n: pd.read_parquet(paths.processed / f"{n}.parquet")  # noqa: E731
    pgf = AM.player_game_frame(r("games"), r("player_games"), r("player_values"))
    pgf = pgf[
        [
            "season",
            "game_id",
            "game_date",
            "player_id",
            "team_id",
            "played",
            "minutes",
            "gp_before",
            "streak_before",
            "V",
            "exp_mp",
            "rotation",
            "y_played",
            "mpg_prev",
            "v_prev",
        ]
    ].copy()
    pgf["game_date"] = pd.to_datetime(pgf.game_date)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pgf.to_parquet(cache, index=False)
    return pgf


def hazards(pgf_train: pd.DataFrame) -> np.ndarray:
    """h[n], n = 0..N_MAX: P(plays the next game | missed n in a row) (h[0] unused)."""
    t = AM.return_table(pgf_train, max_streak=60).set_index("streak_before").p_return_next
    h = np.empty(N_MAX + K_MAX + 2)
    h[0] = 1.0
    for n in range(1, len(h)):
        h[n] = t.get(n, t.iloc[-1])
    return h


def new_injury_curve(pgf_train: pd.DataFrame) -> np.ndarray:
    """pi[k], k = 0..K_MAX: P(misses the team's k-th next game | played the last one),
    rotation players with >= 5 games played, same team stint."""
    x = pgf_train[pgf_train.rotation & (pgf_train.gp_before >= 5)].sort_values(
        ["season", "player_id", "team_id", "game_date"]
    )
    grp = x.groupby(["season", "player_id", "team_id"], sort=False).y_played
    base = x.y_played.to_numpy() == 1
    pi = np.zeros(K_MAX + 1)
    for k in range(1, K_MAX + 1):
        fut = grp.shift(-k).to_numpy()
        ok = base & ~np.isnan(fut)
        pi[k] = float(1 - fut[ok].mean()) if ok.any() else pi[k - 1]
    return pi


def survival(h: np.ndarray, n: int, k_max: int = K_MAX) -> np.ndarray:
    """S[k] = P(still out at the k-th next game | missed n), k = 0..k_max (S[0] = 1)."""
    s = np.ones(k_max + 1)
    if n == 0:
        s[1:] = 0.0
        return s
    n = min(n, N_MAX)
    s[1:] = np.cumprod(1 - h[n : n + k_max])
    return s


def state_at(pgf_season: pd.DataFrame, cutoff: pd.Timestamp, tau_days: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(players, teams) at the cutoff. players: team_id, player_id, V, f, n.
    teams: team_id, rec_v, rec_f."""
    b = pgf_season[pgf_season.game_date < cutoff]
    # rotation players only (availability-model definition, >= 20 mpg this season or last, >= 3 games
    # played for the team): long "absences" of end-of-bench players are mostly DNPs,
    # G League assignments or departures, not injuries
    b = b[b.rotation & (b.gp_before >= 3)]
    if b.empty:
        return (
            pd.DataFrame(columns=["team_id", "player_id", "V", "f", "n"]),
            pd.DataFrame(columns=["team_id", "rec_v", "rec_f"]),
        )
    last = b.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
    players = pd.DataFrame(
        {
            "team_id": last.team_id.astype(int).to_numpy(),
            "player_id": last.player_id.to_numpy(),
            "V": last.V.to_numpy(float),
            "f": (last.exp_mp.to_numpy(float) / 48.0),
            "n": np.where(last.played, 0, last.streak_before + 1).astype(int),
        }
    )
    miss = b.assign(mv=(1 - b.y_played) * b.exp_mp / 48.0 * b.V, mf=(1 - b.y_played) * b.exp_mp / 48.0)
    tg = (
        miss.groupby(["team_id", "game_id"])
        .agg(date=("game_date", "first"), mv=("mv", "sum"), mf=("mf", "sum"))
        .reset_index()
    )
    w = np.exp(-(cutoff - tg.date).dt.days.to_numpy(float) / tau_days)
    tg["w"] = w
    teams = (
        tg.groupby("team_id")
        .apply(
            lambda d: pd.Series({"rec_v": np.sum(d.w * d.mv) / np.sum(d.w), "rec_f": np.sum(d.w * d.mf) / np.sum(d.w)}),
            include_groups=False,
        )
        .reset_index()
    )
    teams["team_id"] = teams.team_id.astype(int)
    return players, teams


def expected_change(
    players: pd.DataFrame, teams: pd.DataFrame, h: np.ndarray, pi: np.ndarray, team_ids: list[int]
) -> dict[int, dict]:
    """team_id -> components for the game model:
      rec_v, rec_f     what the cutoff strength already reflects (scalars)
      ret_v, ret_f     expected missing from players out at the cutoff, at the team's k-th
                       next game (arrays, k = 0..K_MAX)
      new_v, new_f     expected missing from new injuries of healthy players (arrays)
    The game model keeps the components whose out-of-fold value is shown by season-grouped
    k-fold (projection.horizon_cv); new_* is proportional to the healthy roster's value and
    so also proxies team talent, which is why it is not merged into one expected change."""
    rec = teams.set_index("team_id") if len(teams) else None
    out = {}
    for t in team_ids:
        pl = players[players.team_id == t]
        ret_v, ret_f = np.zeros(K_MAX + 1), np.zeros(K_MAX + 1)
        for p in pl[pl.n > 0].itertuples(index=False):
            s = survival(h, int(p.n))
            ret_v += s * p.f * p.V
            ret_f += s * p.f
        ok = pl[pl.n == 0]
        out[t] = {
            "rec_v": float(rec.rec_v.get(t, 0.0)) if rec is not None else 0.0,
            "rec_f": float(rec.rec_f.get(t, 0.0)) if rec is not None else 0.0,
            "ret_v": ret_v,
            "ret_f": ret_f,
            "new_v": pi * float((ok.f * ok.V).sum()),
            "new_f": pi * float(ok.f.sum()),
        }
    return out


def sample_paths(
    players: pd.DataFrame, teams: pd.DataFrame, h: np.ndarray, pi: np.ndarray, team_ids: list[int], n_paths: int, rng
) -> dict[int, dict]:
    """Per team: the expected components of :func:`expected_change` plus, for the players
    out at the cutoff, the sampled number of further games missed R (paths x players) and
    their f V and f, so ret_* can be evaluated per path: ret_v(k) = sum f V [R >= k]."""
    exp = expected_change(players, teams, h, pi, team_ids)
    out = {}
    for t in team_ids:
        hurt = players[(players.team_id == t) & (players.n > 0)]
        R = np.zeros((n_paths, len(hurt)), dtype=int)
        for j, p in enumerate(hurt.itertuples(index=False)):
            s = survival(h, int(p.n))[1:]  # P(still out at game 1..K)
            u = rng.random(n_paths)
            R[:, j] = (u[:, None] < s[None, :]).sum(axis=1)  # games still missed
        out[t] = {**exp[t], "R": R, "fv": (hurt.f * hurt.V).to_numpy(float), "f": hurt.f.to_numpy(float)}
    return out
