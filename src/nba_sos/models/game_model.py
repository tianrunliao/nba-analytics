"""Game model: pre-game single-game prediction, rolling backtest and feature-group ablation.

    python -m nba_sos.pipeline game-model

Target: home margin per 100 possessions (pace-free across eras). Win probability =
Phi(pred / sigma), sigma = residual sd on the last training season (held out from the
fit for early stopping), so no test-season information is used.

Every feature is known before tip-off:
  strength  Elo (pre-game), EWMA net rating (7/14/21 days), dynamic opponent-adjusted
            strength and its sd, previous-season-based pre-season rating, games played
  rest      rest days, back-to-back, games in last 4/7 days, hours since last tip-off,
            overtime in previous game
  travel    distance, time-zone shift, consecutive road games, b2b type, local and
            body-clock tip-off hour
  injury    official pre-game report (2019-20 on): players out/doubtful, their prior
            minutes, and their previous-season player value (LPM); NaN before reports exist

Tuning seasons (``TUNE_SEASONS``) choose Elo and dynamic-strength hyper-parameters and
are never test seasons.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import Paths, season_start_year
from ..evaluation import metrics as M
from ..ratings import dynamic as DY
from ..ratings import elo as E
from . import preseason

log = logging.getLogger(__name__)
TUNE_SEASONS = ["2013-14", "2014-15", "2015-16"]
DYN_GRID = {"tau": (30.0, 60.0, 120.0), "alpha": (2.0, 5.0, 15.0), "rho": (0.5, 0.7, 0.9)}

GROUPS = {
    "strength": [
        "elo_diff",
        "p_home_elo",
        "ewma7_diff",
        "ewma14_diff",
        "ewma21_diff",
        "dyn_diff",
        "dyn_sd_home",
        "dyn_sd_away",
        "prior_diff",
        "gp_home",
        "gp_away",
    ],
    "rest": [
        "rest_home",
        "rest_away",
        "b2b_home",
        "b2b_away",
        "g4_home",
        "g4_away",
        "g7_home",
        "g7_away",
        "turn_home",
        "turn_away",
        "prevot_home",
        "prevot_away",
    ],
    "travel": [
        "km_home",
        "km_away",
        "tz_home",
        "tz_away",
        "road_home",
        "road_away",
        "b2bt_home",
        "b2bt_away",
        "local_hour",
        "body_home",
        "body_away",
    ],
    "injury": [
        "nout_home",
        "nout_away",
        "mmin_home",
        "mmin_away",
        "mval_home",
        "mval_away",
        "inj_ok_home",
        "inj_ok_away",
    ],
}
CATEGORICAL = ["b2bt_home", "b2bt_away"]


# ------------------------------------------------------------------ features
def _tune_dynamic(all_games: pd.DataFrame) -> dict:
    tune = all_games[all_games.season.isin(TUNE_SEASONS)]
    best = None
    sub = all_games[all_games.season.map(season_start_year) <= season_start_year(TUNE_SEASONS[-1])]
    for tau, alpha, rho in itertools.product(DYN_GRID["tau"], DYN_GRID["alpha"], DYN_GRID["rho"]):
        d = DY.dynamic_pregame(sub, tau, alpha, rho)
        x = _side_merge(
            tune[["game_id", "home_team_id", "away_team_id", "net_rating_margin", "is_neutral"]], d, ["dyn_strength"]
        )
        diff = x.dyn_strength_home - x.dyn_strength_away
        hca = float(x[~x.is_neutral].net_rating_margin.mean())
        mse = float(np.mean((x.net_rating_margin - diff - hca * (~x.is_neutral)) ** 2))
        if best is None or mse < best[0]:
            best = (mse, {"tau": tau, "alpha": alpha, "rho": rho})
    return {**best[1], "tune_mse": best[0]}


def _side_merge(base: pd.DataFrame, per_team: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Attach per-(game, team) columns for the home and away team as <col>_home / <col>_away."""
    out = base.copy()
    out["home_team_id"], out["away_team_id"] = out.home_team_id.astype(int), out.away_team_id.astype(int)
    pt = per_team[["game_id", "team_id"] + cols].copy()
    pt["team_id"] = pt.team_id.astype(int)
    for side in ("home", "away"):
        p = pt.rename(columns={"team_id": f"{side}_team_id", **{c: f"{c}_{side}" for c in cols}})
        out = out.merge(p, on=["game_id", f"{side}_team_id"], how="left")
    return out


def _injury_value(availability: pd.DataFrame, player_values: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: sum over Out/Doubtful players of previous-season LPM * prior mpg/48."""
    a = availability[availability.injury_status.isin(["Out", "Doubtful"]) & availability.player_id.notna()].copy()
    prev = player_values[["season", "player_id", "lpm", "min_total", "gp"]].copy()
    prev["season"] = prev.season.map(lambda s: f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}")
    prev["mpg_prev"] = prev.min_total / prev.gp.replace(0, np.nan)
    a = a.merge(prev[["season", "player_id", "lpm", "mpg_prev"]], on=["season", "player_id"], how="left")
    a["v"] = a.lpm.fillna(0) * a.mpg_prev.fillna(0) / 48
    v = a.groupby(["game_id", "team_id"]).v.sum().rename("missing_value_prev").reset_index()
    v["team_id"] = v.team_id.astype(int)
    f = features[["game_id", "team_id", "injury_snapshot_usable"]].assign(team_id=lambda x: x.team_id.astype(int))
    f = f.merge(v, on=["game_id", "team_id"], how="left")
    f["missing_value_prev"] = f.missing_value_prev.fillna(0.0).where(f.injury_snapshot_usable.astype(bool))
    return f[["game_id", "team_id", "missing_value_prev"]]


def build_features(paths: Paths) -> tuple[pd.DataFrame, dict]:
    P_ = paths.processed
    games = pd.read_parquet(P_ / "games.parquet")
    prior_games = pd.read_parquet(P_ / "prior_season_games.parquet")
    all_games = pd.concat([prior_games, games], ignore_index=True)
    feats = pd.read_parquet(P_ / "team_game_features.parquet")
    avail = pd.read_parquet(P_ / "player_availability.parquet")
    pv = pd.read_parquet(P_ / "player_values.parquet")

    elo_par = E.tune(
        all_games,
        [s for s in all_games.season.unique() if season_start_year(s) <= season_start_year(TUNE_SEASONS[-1])],
        burn_in=[all_games.season.min()],
    )
    elo = E.run_elo(all_games, elo_par["k"], elo_par["hca"], elo_par["carry"])
    dyn_par = _tune_dynamic(all_games)
    dyn = DY.dynamic_pregame(all_games, dyn_par["tau"], dyn_par["alpha"], dyn_par["rho"])
    ew = {t: DY.ewma_pregame(all_games, t).rename(columns={"ewma_net": f"ewma{t}"}) for t in (7, 14, 21)}
    log.info("elo %s dyn %s", elo_par, dyn_par)

    base = games[
        [
            "season",
            "game_id",
            "game_date",
            "tipoff_utc",
            "home_team_id",
            "away_team_id",
            "is_neutral",
            "net_rating_margin",
            "home_win",
            "margin",
        ]
    ].copy()
    base["home_team_id"], base["away_team_id"] = base.home_team_id.astype(int), base.away_team_id.astype(int)
    x = base.merge(elo, on="game_id", how="left")
    x = _side_merge(x, dyn, ["dyn_strength", "dyn_sd"])
    for t, e in ew.items():
        x = _side_merge(x, e, [f"ewma{t}"])
    tf = feats.assign(team_id=feats.team_id.astype(int))
    tf = tf.merge(_injury_value(avail, pv, feats), on=["game_id", "team_id"], how="left")
    cols = {
        "season_games_played": "gp",
        "rest_days": "rest",
        "is_back_to_back": "b2b",
        "games_last_4_days": "g4",
        "games_last_7_days": "g7",
        "turnaround_hours": "turn",
        "prev_overtime_periods": "prevot",
        "travel_distance_km": "km",
        "timezone_change_hours": "tz",
        "consecutive_road_games": "road",
        "b2b_type": "b2bt",
        "body_clock_tipoff_hour": "body",
        "n_out": "nout",
        "missing_minutes_prior_avg": "mmin",
        "missing_value_prev": "mval",
        "injury_snapshot_usable": "inj_ok",
        "tipoff_local_hour": "lh",
    }
    t2 = tf[["game_id", "team_id"] + list(cols)].rename(columns=cols)
    x = _side_merge(x, t2, list(cols.values()))
    pr = preseason.prior_ratings(all_games)
    x = x.merge(
        pr.rename(columns={"team_id": "home_team_id", "prior_net_rating": "prior_h"}),
        on=["season", "home_team_id"],
        how="left",
    )
    x = x.merge(
        pr.rename(columns={"team_id": "away_team_id", "prior_net_rating": "prior_a"}),
        on=["season", "away_team_id"],
        how="left",
    )
    x["elo_diff"] = x.home_elo - x.away_elo
    x["dyn_diff"] = x.dyn_strength_home - x.dyn_strength_away
    x["dyn_sd_home"], x["dyn_sd_away"] = x.dyn_sd_home, x.dyn_sd_away
    for t in (7, 14, 21):
        x[f"ewma{t}_diff"] = x[f"ewma{t}_home"] - x[f"ewma{t}_away"]
    x["prior_diff"] = x.prior_h - x.prior_a
    x["local_hour"] = x.lh_home
    x["home_flag"] = (~x.is_neutral).astype(float)
    for side in ("home", "away"):
        x[f"b2b_{side}"] = x[f"b2b_{side}"].astype("boolean").astype(float)
        x[f"inj_ok_{side}"] = x[f"inj_ok_{side}"].astype("boolean").astype(float)
        x[f"b2bt_{side}"] = x[f"b2bt_{side}"].fillna("none").astype(str)
    x["y"] = x.net_rating_margin.astype(float)
    return x.sort_values(["tipoff_utc", "game_id"]).reset_index(drop=True), {"elo": elo_par, "dynamic": dyn_par}


# ------------------------------------------------------------------ models
def _fit_catboost(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], seed: int = 0):
    from catboost import CatBoostRegressor, Pool

    cats = [c for c in cols if c in CATEGORICAL]
    m = CatBoostRegressor(
        iterations=2000,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        od_type="Iter",
        od_wait=100,
        allow_writing_files=False,
    )
    m.fit(
        Pool(train[cols], train.y, cat_features=cats),
        eval_set=Pool(valid[cols], valid.y, cat_features=cats),
        use_best_model=True,
    )
    sigma = float(np.std(valid.y - m.predict(valid[cols])))
    return m, sigma


def _platt(margin: np.ndarray, won: np.ndarray) -> tuple[float, float]:
    """Logistic calibration p = 1/(1+exp(-(a*margin + b))) by Newton's method."""
    a, b = 0.1, 0.0
    for _ in range(50):
        z = np.clip(a * margin + b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((p - won) * margin), np.sum(p - won)])
        wgt = p * (1 - p)
        H = np.array([[np.sum(wgt * margin**2), np.sum(wgt * margin)], [np.sum(wgt * margin), np.sum(wgt)]])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def _linear(train: pd.DataFrame, cols: list[str]):
    X = np.column_stack([train[c].fillna(0).to_numpy(float) for c in cols])
    b, *_ = np.linalg.lstsq(X, train.y.to_numpy(float), rcond=None)
    sigma = float(np.std(train.y.to_numpy(float) - X @ b))
    return b, sigma


def _metrics(y_win, y, pred, sigma) -> dict:
    p = norm.cdf(pred / sigma)
    return {
        "log_loss": M.log_loss(y_win, p),
        "brier": M.brier(y_win, p),
        "mae_per100": M.mae(y, pred),
        "ece": M.expected_calibration_error(y_win, p),
    }


def backtest(x: pd.DataFrame, test_seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hist = sorted(x.season.unique(), key=season_start_year)
    first = hist.index(TUNE_SEASONS[0])
    specs = {
        "cb_strength": GROUPS["strength"] + ["home_flag"],
        "cb_+rest": GROUPS["strength"] + GROUPS["rest"] + ["home_flag"],
        "cb_+rest+travel": GROUPS["strength"] + GROUPS["rest"] + GROUPS["travel"] + ["home_flag"],
        "cb_full": GROUPS["strength"] + GROUPS["rest"] + GROUPS["travel"] + GROUPS["injury"] + ["home_flag"],
    }
    rows, preds, importances = [], [], {}
    for test in test_seasons:
        train_seasons = [s for s in hist[first:] if season_start_year(s) < season_start_year(test)]
        tr_all = x[x.season.isin(train_seasons)]
        valid = tr_all[tr_all.season == train_seasons[-1]]
        tr = tr_all[tr_all.season != train_seasons[-1]]
        te = x[x.season == test]
        yw, y = te.home_win.astype(float).to_numpy(), te.y.to_numpy(float)
        res = {"season": test, "n_games": len(te), "train_seasons": len(train_seasons)}
        # simple baselines (fit on all training seasons)
        p_elo = te.p_home_elo.to_numpy(float)
        res.update(
            {
                f"elo_{k}": v
                for k, v in {
                    "log_loss": M.log_loss(yw, p_elo),
                    "brier": M.brier(yw, p_elo),
                    "ece": M.expected_calibration_error(yw, p_elo),
                }.items()
            }
        )
        for name, cols in {
            "preseason": ["prior_diff", "home_flag"],
            "ewma14": ["ewma14_diff", "home_flag"],
            "dynamic": ["dyn_diff", "home_flag"],
        }.items():
            b, sg = _linear(tr_all, cols)
            pred = np.column_stack([te[c].fillna(0).to_numpy(float) for c in cols]) @ b
            res.update({f"{name}_{k}": v for k, v in _metrics(yw, y, pred, sg).items()})
        for name, cols in specs.items():
            m, sg = _fit_catboost(tr, valid, cols)
            pred = m.predict(te[cols])
            res.update({f"{name}_{k}": v for k, v in _metrics(yw, y, pred, sg).items()})
            if name == "cb_full":
                # probability calibration fit on the held-out last training season only
                a, b = _platt(m.predict(valid[cols]), valid.home_win.astype(float).to_numpy())
                p_cal = 1 / (1 + np.exp(-(a * pred + b)))
                res.update(
                    {
                        "cb_full_cal_log_loss": M.log_loss(yw, p_cal),
                        "cb_full_cal_brier": M.brier(yw, p_cal),
                        "cb_full_cal_mae_per100": M.mae(y, pred),
                        "cb_full_cal_ece": M.expected_calibration_error(yw, p_cal),
                        "platt_a": a,
                        "platt_b": b,
                    }
                )
                preds.append(
                    te[["season", "game_id", "home_team_id", "away_team_id", "y", "home_win"]].assign(
                        pred_margin100=pred, p_home=norm.cdf(pred / sg), p_home_cal=p_cal
                    )
                )
                importances[test] = dict(zip(cols, m.get_feature_importance()))
        rows.append(res)
        log.info(
            "game-model %s: elo %.4f dyn %.4f cb_full %.4f",
            test,
            res["elo_log_loss"],
            res["dynamic_log_loss"],
            res["cb_full_log_loss"],
        )
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), importances


def pooled(m: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    w = m.n_games
    rows = []
    for mod in models:
        r = {"model": mod}
        for k in ("log_loss", "brier", "mae_per100", "ece"):
            col = f"{mod}_{k}"
            if col in m:
                r[k] = float((m[col] * w).sum() / w.sum())
        rows.append(r)
    return pd.DataFrame(rows)
