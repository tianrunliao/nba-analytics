"""Phase 1 baseline: pre-season rating + home-court advantage.

Pre-season rating of team i in season s = its previous-season net rating per 100
possessions (decision #1). Margin model, fit by OLS on training seasons only:

    margin_g = b * (prior_home - prior_away) + HCA * [not neutral] + e_g

``b`` both converts per-100 units to points and shrinks last season toward the mean
(b < pace/100 means regression to 0). Win probability = Phi(margin_hat / sigma), with
sigma the training residual standard deviation. A constant baseline (training home
win rate / mean margin; 0.5 and 0 at neutral sites) is reported alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import previous_season
from ..evaluation import metrics as M
from ..features.pregame import team_game_long
from ..sos.static import season_team_totals


def prior_ratings(all_games: pd.DataFrame) -> pd.DataFrame:
    """season, team_id, prior_net_rating (from the previous season's games only)."""
    tot = season_team_totals(team_game_long(all_games))
    tot["season"] = tot.season.map(lambda s: f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}")
    return tot.rename(columns={"net_rating": "prior_net_rating"})[["season", "team_id", "prior_net_rating"]]


def design(games: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    p = priors.rename(columns={"team_id": "home_team_id", "prior_net_rating": "prior_home"})
    q = priors.rename(columns={"team_id": "away_team_id", "prior_net_rating": "prior_away"})
    d = games.merge(p, on=["season", "home_team_id"], how="left").merge(q, on=["season", "away_team_id"], how="left")
    d["prior_diff"] = d.prior_home - d.prior_away
    d["home_flag"] = (~d.is_neutral).astype(float)
    return d


def fit(train: pd.DataFrame) -> dict:
    if train[["prior_diff"]].isna().any().any():
        raise ValueError("training rows without a prior rating")
    X = train[["prior_diff", "home_flag"]].to_numpy(float)
    y = train.margin.to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))
    nn = train[~train.is_neutral]
    return {
        "b": float(coef[0]),
        "hca": float(coef[1]),
        "sigma": sigma,
        "n_train": int(len(y)),
        "const_home_win_rate": float(nn.home_win.mean()),
        "const_home_margin": float(nn.margin.mean()),
    }


def predict(params: dict, d: pd.DataFrame) -> pd.DataFrame:
    out = d[
        [
            "season",
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "is_neutral",
            "margin",
            "home_win",
            "prior_home",
            "prior_away",
        ]
    ].copy()
    out["pred_margin"] = params["b"] * d.prior_diff + params["hca"] * d.home_flag
    out["pred_p_home"] = norm.cdf(out.pred_margin / params["sigma"])
    out["const_margin"] = np.where(d.is_neutral, 0.0, params["const_home_margin"])
    out["const_p_home"] = np.where(d.is_neutral, 0.5, params["const_home_win_rate"])
    return out


def evaluate(pred: pd.DataFrame) -> dict:
    y, m = pred.home_win.to_numpy(float), pred.margin.to_numpy(float)
    res: dict[str, float] = {"n_games": int(len(pred))}
    for name, pc, mc in [("preseason", "pred_p_home", "pred_margin"), ("constant", "const_p_home", "const_margin")]:
        res[f"{name}_log_loss"] = M.log_loss(y, pred[pc])
        res[f"{name}_brier"] = M.brier(y, pred[pc])
        res[f"{name}_margin_mae"] = M.mae(m, pred[mc])
        res[f"{name}_margin_rmse"] = M.rmse(m, pred[mc])
        res[f"{name}_ece"] = M.expected_calibration_error(y, pred[pc].to_numpy(float))
    return res


def rolling_backtest(
    games: pd.DataFrame, all_games: pd.DataFrame, model_cfg: dict, historical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``games``: historical seasons; ``all_games``: also includes prior-only seasons."""
    cfg = model_cfg["baseline"]
    d = design(games, prior_ratings(all_games))
    preds, rows = [], []
    for train_seasons, test in M.rolling_splits(historical, cfg["first_train_season"], cfg["test_seasons"]):
        tr, te = d[d.season.isin(train_seasons)], d[d.season == test]
        assert previous_season(test) in set(all_games.season), f"no prior season for {test}"
        params = fit(tr)
        p = predict(params, te)
        preds.append(p)
        rows.append({"test_season": test, "train_seasons": ",".join(train_seasons), **params, **evaluate(p)})
    pred = pd.concat(preds, ignore_index=True)
    summary = pd.DataFrame(rows)
    pooled = {"test_season": "pooled", "train_seasons": "", **evaluate(pred)}
    return pred, pd.concat([summary, pd.DataFrame([pooled])], ignore_index=True)
