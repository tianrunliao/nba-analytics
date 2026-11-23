"""Schedule difficulty over a window of each team's games (e.g. its first 15).

Interim strength estimate (to be replaced by the Phase 2 dynamic model):

    S_i = a * prior_net_rating_i + c * net_rating_first_n_i

fit on history by OLS against each team's rest-of-season net rating, where the
"first n" are the team's first n games. Game model, fit on the rest-of-season games of
the same history (reuses the Phase 1 baseline fitter):

    margin_home = g * (S_home - S_away) + HCA * [not neutral] + e,   e ~ N(0, sigma)

D-SOS (plan section 9): an average team (S = 0) plays the window with the real
home/away/neutral pattern; EW_ref = sum of its win probabilities;
D-SOS = n/2 - EW_ref (positive = harder). Opponent-only SOS drops home court.
Fatigue / travel are reported as separate burden columns; their effect sizes are not
yet estimated well enough (Phase 2) to be folded into D-SOS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..features.pregame import team_game_long
from ..models import preseason


def first_n_stats(t: pd.DataFrame, n: int) -> pd.DataFrame:
    """Net rating over each team-season's first n completed games, and over the rest."""
    t = t.sort_values(["season", "team_id", "tipoff_utc"]).copy()
    t["k"] = t.groupby(["season", "team_id"]).cumcount()
    def net(d):
        return (d.pts_for.sum() - d.pts_against.sum()) / d.possessions.sum() * 100
    first = t[t.k < n].groupby(["season", "team_id"]).apply(net, include_groups=False).rename("net_first_n")
    rest = t[t.k >= n].groupby(["season", "team_id"]).apply(net, include_groups=False).rename("net_rest")
    return pd.concat([first, rest], axis=1).reset_index()


def fit_strength(games_hist: pd.DataFrame, all_games: pd.DataFrame, n: int) -> dict:
    t = team_game_long(games_hist)
    d = first_n_stats(t, n).merge(preseason.prior_ratings(all_games), on=["season", "team_id"])
    d = d.dropna()
    X = d[["prior_net_rating", "net_first_n"]].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, d.net_rest.to_numpy(float), rcond=None)
    a, c = map(float, coef)
    d["S"] = a * d.prior_net_rating + c * d.net_first_n
    # game model on rest-of-season games, strengths frozen at the n-game snapshot
    t["k"] = t.sort_values(["season", "team_id", "tipoff_utc"]).groupby(["season", "team_id"]).cumcount()
    home_k = t[t.home_away == "H"][["game_id", "k"]].rename(columns={"k": "k_home"})
    away_k = t[t.home_away == "A"][["game_id", "k"]].rename(columns={"k": "k_away"})
    gm = games_hist.merge(home_k, on="game_id").merge(away_k, on="game_id")
    gm = gm[(gm.k_home >= n) & (gm.k_away >= n)]
    s = d[["season", "team_id", "S"]]
    gm = gm.merge(s.rename(columns={"team_id": "home_team_id", "S": "S_home"}), on=["season", "home_team_id"]) \
           .merge(s.rename(columns={"team_id": "away_team_id", "S": "S_away"}), on=["season", "away_team_id"])
    gm["prior_diff"] = gm.S_home - gm.S_away
    gm["home_flag"] = (~gm.is_neutral).astype(float)
    p = preseason.fit(gm)
    return {"a": a, "c": c, "n_team_seasons": int(len(d)), "g": p["b"], "hca": p["hca"],
            "sigma": p["sigma"], "n_games": p["n_train"],
            "strength_r2": float(1 - ((d.net_rest - d.S) ** 2).sum() / ((d.net_rest - d.net_rest.mean()) ** 2).sum())}


def reference_win_prob(opp_strength: np.ndarray, venue: np.ndarray, params: dict, home_court: bool = True) -> np.ndarray:
    """P(average team wins). venue: +1 average team at home, -1 away, 0 neutral."""
    margin = -params["g"] * np.asarray(opp_strength, float)
    if home_court:
        margin = margin + params["hca"] * np.asarray(venue, float)
    return norm.cdf(margin / params["sigma"])


def window_sos(window: pd.DataFrame, strengths: pd.DataFrame, params: dict) -> pd.DataFrame:
    """``window``: one row per team-game (team_id, opponent_id, home_away, is_neutral, ...).
    ``strengths``: either (team_id, S) — one strength per opponent — or
    (team_id, opponent_id, S) — opponent strength as seen by each team, e.g. excluding
    the opponent's games against that team."""
    if "opponent_id" in strengths.columns:
        w = window.merge(strengths.rename(columns={"S": "opp_S"}), on=["team_id", "opponent_id"], how="left")
    else:
        w = window.merge(strengths.rename(columns={"team_id": "opponent_id", "S": "opp_S"}), on="opponent_id", how="left")
    if w.opp_S.isna().any():
        raise ValueError("opponent without a strength estimate")
    venue = np.where(w.is_neutral, 0, np.where(w.home_away == "H", 1, -1))
    w["p_ref"] = reference_win_prob(w.opp_S, venue, params, home_court=True)
    w["p_ref_neutral"] = reference_win_prob(w.opp_S, venue, params, home_court=False)
    out = w.groupby(["team_id", "team"]).agg(
        n=("game_id", "size"), ew_ref=("p_ref", "sum"), ew_opp_only=("p_ref_neutral", "sum"),
        opp_strength_mean=("opp_S", "mean")).reset_index()
    out["d_sos"] = out.n / 2 - out.ew_ref
    out["opp_sos"] = out.n / 2 - out.ew_opp_only
    out["home_court_part"] = out.d_sos - out.opp_sos
    out["d_sos_rank"] = out.d_sos.rank(ascending=False, method="min").astype(int)
    return out.sort_values("d_sos_rank").reset_index(drop=True)
