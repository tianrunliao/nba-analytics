"""Schedule-, venue-, rest- and health-adjusted regular-season team strength.

Pooled weighted ridge regression over regular-season games of several seasons. Target is
the per-100-possession margin (net_rating_margin), which removes the pace trend across
seasons (91.6 possessions per game in 1996-97, ~100 now):

    y_g = theta[home, s] - theta[away, s] + HCA_s * [not neutral]
          + b_b2b * (b2b_home - b2b_away)
          + b_min * (hi_minutes_home - hi_minutes_away)
          + b_val * (hi_value_home - hi_value_away) + e_g

theta[i, s] is the team's strength in season s with its full roster available, against a
league-average opponent on a neutral court (so schedule difficulty is removed). Each
season has its own home-court advantage HCA_s (year-specific standardisation). Ridge
penalty ``alpha`` on theta only: a N(0, sigma^2/alpha) prior on team strength.

Momentum: within one season, theta is refit with exponentially decaying weights
exp(-(days before season end)/tau) and a ridge prior centred on the full-season theta, with
the shared coefficients held fixed; ``momentum = theta_recent - theta_full``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 10.0  # ~ sigma 15 / prior sd 4.7 per 100 possessions


def _design(
    df: pd.DataFrame, team_index: dict, seasons: list[str], use_health: bool = True
) -> tuple[np.ndarray, list[str]]:
    n, k = len(df), len(team_index)
    X = np.zeros((n, k + len(seasons) + 3))
    hi = df.home_key.map(team_index).to_numpy()
    ai = df.away_key.map(team_index).to_numpy()
    X[np.arange(n), hi] = 1.0
    X[np.arange(n), ai] = -1.0
    sidx = df.season.map({s: i for i, s in enumerate(seasons)}).to_numpy()
    X[np.arange(n), k + sidx] = df.home_flag.to_numpy(float)
    X[:, k + len(seasons)] = df.d_b2b.to_numpy(float)
    X[:, k + len(seasons) + 1] = df.d_hi_minutes.to_numpy(float)
    X[:, k + len(seasons) + 2] = df.d_hi_value.to_numpy(float)
    names = [f"hca_{s}" for s in seasons] + ["b_b2b", "b_min", "b_val"]
    if not use_health:
        return X[:, :-2], names[:-2]
    return X, names


def game_frame(games: pd.DataFrame, features: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
    """Join per-game inputs. ``features``: team_game_features (for is_back_to_back)."""
    b2b = features[["game_id", "team_id", "is_back_to_back"]].copy()
    b2b["b2b"] = b2b.is_back_to_back.astype("boolean").fillna(False).astype(float)
    h = health.copy()
    d = games[
        ["season", "game_id", "game_date", "home_team_id", "away_team_id", "is_neutral", "net_rating_margin"]
    ].copy()
    for side in ("home", "away"):
        d = d.merge(
            b2b[["game_id", "team_id", "b2b"]].rename(columns={"team_id": f"{side}_team_id", "b2b": f"{side}_b2b"}),
            on=["game_id", f"{side}_team_id"],
            how="left",
        )
        d = d.merge(
            h[["game_id", "team_id", "hi_minutes", "hi_value"]].rename(
                columns={
                    "team_id": f"{side}_team_id",
                    "hi_minutes": f"{side}_hi_minutes",
                    "hi_value": f"{side}_hi_value",
                }
            ),
            on=["game_id", f"{side}_team_id"],
            how="left",
        )
    for c in ["home_b2b", "away_b2b", "home_hi_minutes", "away_hi_minutes", "home_hi_value", "away_hi_value"]:
        d[c] = d[c].fillna(0.0)
    d["d_b2b"] = d.home_b2b - d.away_b2b
    d["d_hi_minutes"] = d.home_hi_minutes - d.away_hi_minutes
    d["d_hi_value"] = d.home_hi_value - d.away_hi_value
    d["home_flag"] = (~d.is_neutral).astype(float)
    d["home_key"] = d.season + "|" + d.home_team_id.astype(str)
    d["away_key"] = d.season + "|" + d.away_team_id.astype(str)
    d["y"] = d.net_rating_margin.astype(float)
    return d


def fit_pooled(
    d: pd.DataFrame,
    alpha: float = ALPHA,
    weights: np.ndarray | None = None,
    use_health: bool = True,
    heteroscedastic: bool = False,
) -> dict:
    """``heteroscedastic``: second pass weighting each game by 1/sigma_s^2, where sigma_s is
    that season's first-pass residual sd (game noise rose ~11% from 2013-14 to 2025-26 as
    three-point volume grew, so noisier seasons should count less for shared coefficients)."""
    keys = sorted(set(d.home_key) | set(d.away_key))
    team_index = {k: i for i, k in enumerate(keys)}
    seasons = sorted(d.season.unique())
    X, names = _design(d, team_index, seasons, use_health)
    y = d.y.to_numpy(float)
    pen = np.zeros(X.shape[1])
    pen[: len(keys)] = alpha

    def solve(w):
        XtW = X.T * w
        return np.linalg.solve(XtW @ X + np.diag(pen), XtW @ y)

    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    beta = solve(w)
    resid = y - X @ beta
    season_sigma = pd.Series(resid).groupby(d.season.to_numpy()).apply(lambda r: float(np.sqrt(np.mean(r**2))))
    if heteroscedastic:
        sw = d.season.map(1 / season_sigma**2).to_numpy(float)
        w = w * sw / sw.mean()
        beta = solve(w)
        resid = y - X @ beta
        season_sigma = pd.Series(resid).groupby(d.season.to_numpy()).apply(lambda r: float(np.sqrt(np.mean(r**2))))
    theta = pd.DataFrame({"key": keys, "theta": beta[: len(keys)]})
    theta[["season", "team_id"]] = theta.key.str.split("|", expand=True)
    theta["team_id"] = theta.team_id.astype(int)
    coefs = dict(zip(names, beta[len(keys) :]))
    for c in ("b_min", "b_val"):
        coefs.setdefault(c, 0.0)
    return {
        "theta": theta.drop(columns="key"),
        "coefs": coefs,
        "season_sigma": season_sigma.to_dict(),
        "sigma": float(np.sqrt(np.average(resid**2, weights=w))),
    }


def fit_recent(d_season: pd.DataFrame, fit: dict, tau_days: float, alpha: float = ALPHA) -> pd.DataFrame:
    """Recency-weighted theta for one season, shrunk toward the full-season theta."""
    season = d_season.season.iloc[0]
    c = fit["coefs"]
    full = fit["theta"][fit["theta"].season == season].set_index("team_id").theta
    y = (
        d_season.y
        - c[f"hca_{season}"] * d_season.home_flag
        - c["b_b2b"] * d_season.d_b2b
        - c["b_min"] * d_season.d_hi_minutes
        - c["b_val"] * d_season.d_hi_value
    ).to_numpy(float)
    teams = sorted(full.index)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(d_season)
    X = np.zeros((n, len(teams)))
    X[np.arange(n), d_season.home_team_id.astype(int).map(idx).to_numpy()] = 1.0
    X[np.arange(n), d_season.away_team_id.astype(int).map(idx).to_numpy()] = -1.0
    dates = pd.to_datetime(d_season.game_date)
    age = (dates.max() - dates).dt.days.to_numpy(float)
    w = np.exp(-age / tau_days)
    prior = full.reindex(teams).to_numpy(float)
    XtW = X.T * w
    A = XtW @ X + alpha * np.eye(len(teams))
    b = XtW @ y + alpha * prior
    rec = np.linalg.solve(A, b)
    return pd.DataFrame(
        {"season": season, "team_id": teams, "theta_full": prior, "theta_recent": rec, "momentum": rec - prior}
    )
