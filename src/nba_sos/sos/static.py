"""Static opponent-strength SOS (decision #2).

retro_*  : opponents' FINAL season record / net rating. Descriptive only — uses
           information from after each game and must never feed a prediction.
causal_* : for each game, the opponent's record / net rating over games on dates
           strictly before that game; season value = mean over the team's games.
Opponent win% excludes the opponent's games against the team itself (standard OWP);
opponent net rating does not.
Ranks: 1 = hardest schedule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.pregame import team_game_long


def season_team_totals(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["season", "team_id"])
    return pd.DataFrame(
        {
            "games": g.size(),
            "wins": g.win.sum(),
            "net_rating": (g.pts_for.sum() - g.pts_against.sum()) / g.possessions.sum() * 100,
        }
    ).reset_index()


def retro_sos(t: pd.DataFrame) -> pd.DataFrame:
    tot = season_team_totals(t)
    # opponent record vs everyone except team i
    h2h = t.groupby(["season", "team_id", "opponent_id"]).agg(g=("win", "size"), w=("win", "sum")).reset_index()
    # opponent's results in games against i = (games, wins) from the opponent's perspective
    opp_vs_i = h2h.rename(columns={"team_id": "opp", "opponent_id": "team_id", "g": "g_vs_i", "w": "w_vs_i"})
    x = (
        t[["season", "team_id", "opponent_id"]]
        .merge(
            tot.rename(columns={"team_id": "opponent_id", "games": "og", "wins": "ow", "net_rating": "onr"}),
            on=["season", "opponent_id"],
        )
        .merge(opp_vs_i, left_on=["season", "team_id", "opponent_id"], right_on=["season", "team_id", "opp"])
    )
    x["owp"] = (x.ow - x.w_vs_i) / (x.og - x.g_vs_i)
    r = (
        x.groupby(["season", "team_id"])
        .agg(retro_owp=("owp", "mean"), retro_opp_net_rating=("onr", "mean"))
        .reset_index()
    )
    return r


def causal_per_game(t: pd.DataFrame) -> pd.DataFrame:
    """Opponent's pre-game record (excluding games vs this team) and net rating."""
    t = t.sort_values(["season", "team_id", "game_date"]).copy()
    rows = []
    for season, ts in t.groupby("season"):
        ts = ts.reset_index(drop=True)
        by_team = {tid: grp.sort_values("game_date") for tid, grp in ts.groupby("team_id")}
        for r in ts.itertuples():
            o = by_team[r.opponent_id]
            prev = o[o.game_date < r.game_date]
            prev_ex = prev[prev.opponent_id != r.team_id]
            n_ex = len(prev_ex)
            rows.append(
                {
                    "season": season,
                    "game_id": r.game_id,
                    "team_id": r.team_id,
                    "opponent_id": r.opponent_id,
                    "opp_games_to_date_ex": n_ex,
                    "opp_win_pct_to_date_ex": prev_ex.win.mean() if n_ex else np.nan,
                    "opp_net_rating_to_date": (
                        (prev.pts_for.sum() - prev.pts_against.sum()) / prev.possessions.sum() * 100
                    )
                    if len(prev)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def static_sos(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = team_game_long(games)
    per_game = causal_per_game(t)
    causal = (
        per_game.groupby(["season", "team_id"])
        .agg(
            causal_owp_mean=("opp_win_pct_to_date_ex", "mean"),
            causal_opp_net_rating_mean=("opp_net_rating_to_date", "mean"),
            causal_games_with_value=("opp_win_pct_to_date_ex", "count"),
        )
        .reset_index()
    )
    tot = season_team_totals(t)
    out = tot.merge(retro_sos(t), on=["season", "team_id"]).merge(causal, on=["season", "team_id"])
    out = out.merge(t.drop_duplicates(["season", "team_id"])[["season", "team_id", "team"]], on=["season", "team_id"])
    for c in ["retro_owp", "retro_opp_net_rating", "causal_owp_mean", "causal_opp_net_rating_mean"]:
        out[f"{c}_rank"] = out.groupby("season")[c].rank(ascending=False, method="min").astype(int)
    out["win_pct"] = out.wins / out.games
    return out.sort_values(["season", "retro_owp_rank"]).reset_index(drop=True), per_game
