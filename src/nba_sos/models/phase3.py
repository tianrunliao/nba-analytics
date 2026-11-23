"""Phase 3: player availability, expected minutes, lineup contribution, double-counting
check, injury-module evaluation and return-time (survival) tables.

    python -m nba_sos.pipeline phase3

Pre-game quantities for player p on team i before game g (all causal):
  q_p     P(plays) from the availability model: report status x rotation role x reason
          category x consecutive games already missed (model fit on earlier seasons only)
  E[MP]   mean minutes over his last 10 games played this season (else season-to-date,
          else previous season), as a share of 48
  V_p     previous-season player value (LPM), shrunk by previous-season minutes; players
          without a previous NBA season get 0 (league average) — flagged in the report

Lineup contribution (plan 7.1):
  M_today  = sum_p (1 - q_p) * E[MP]/48 * V_p      expected value missing today
  M_recent = time-decayed (tau = dynamic-strength tau) mean of the value actually missing
             in the team's earlier games this season — what the dynamic strength has
             already absorbed
  dM       = M_today - M_recent                    the part NOT yet in the strength rating
Double counting: if the dynamic rating already reflects long absences, y should load on
dM, and adding M_recent separately should carry a coefficient of about -coef(M_today).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import Paths, season_start_year
from ..evaluation import metrics as M
from ..ratings import players as P

log = logging.getLogger(__name__)
ROTATION_MPG = 20.0
STREAK_BINS = [-1, 0, 1, 3, 7, 15, 1000]
STREAK_LABELS = ["0", "1", "2-3", "4-7", "8-15", "16+"]

REASON_RULES = [  # (category, keywords in lower-cased, space-free reason)
    ("g_league", ["gleague"]),
    ("health_protocol", ["healthandsafety", "covid", "concussionprotocol"]),
    ("not_with_team", ["notwithteam", "tradepending", "ineligible"]),
    ("personal", ["personal"]),
    ("suspension", ["suspension"]),
    ("reconditioning", ["reconditioning", "returntocompetition"]),
    ("rest", ["rest"]),
    ("injury_management", ["management"]),
    ("injury", ["injury", "illness"]),
]


def reason_category(reason: str | float) -> str:
    if not isinstance(reason, str) or not reason:
        return "unknown"
    r = reason.lower().replace(" ", "")
    for cat, kws in REASON_RULES:
        if any(k in r for k in kws):
            return cat
    return "other"


# ------------------------------------------------------------------ player-game frame
def player_game_frame(games: pd.DataFrame, player_games: pd.DataFrame, player_values: pd.DataFrame) -> pd.DataFrame:
    """One row per (roster player, team game) with causal features and the outcome."""
    rg = P.roster_games(player_games, games)
    rg = rg.sort_values(["season", "player_id", "team_id", "game_date", "game_id"]).reset_index(drop=True)
    grp = rg.groupby(["season", "player_id", "team_id"], sort=False)
    played = rg.played.astype(float)
    rg["gp_before"] = grp.played.transform(lambda s: s.astype(int).cumsum().shift(1).fillna(0))
    cum_min = grp.minutes.transform(lambda s: s.cumsum().shift(1).fillna(0))
    rg["mpg_to_date"] = (cum_min / rg.gp_before.replace(0, np.nan))
    # mean of the last 10 games PLAYED strictly before this game: rolling mean on played
    # rows (inclusive), then shifted by one row and carried forward over missed games
    keys = [rg.season, rg.player_id, rg.team_id]
    pl = rg[rg.played]
    roll = pl.minutes.groupby([pl.season, pl.player_id, pl.team_id]).transform(
        lambda s: s.rolling(10, min_periods=1).mean())
    incl = pd.Series(np.nan, index=rg.index)
    incl[pl.index] = roll
    rg["mpg_last10"] = incl.groupby(keys).transform(lambda s: s.shift(1).ffill())

    def streak(s: pd.Series) -> pd.Series:
        out, run = [], 0
        for p in s:
            out.append(run)
            run = 0 if p else run + 1
        return pd.Series(out, index=s.index)
    rg["streak_before"] = grp.played.transform(streak)
    # previous-season value and minutes
    prev = player_values[["season", "player_id", "lpm", "min_total", "gp"]].copy()
    prev["season"] = prev.season.map(lambda s: f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}")
    prev = prev.rename(columns={"lpm": "v_prev", "min_total": "min_prev", "gp": "gp_prev"})
    rg = rg.merge(prev, on=["season", "player_id"], how="left")
    rg["mpg_prev"] = rg.min_prev / rg.gp_prev.replace(0, np.nan)
    rg["has_prev"] = rg.v_prev.notna()
    rg["V"] = (rg.v_prev * rg.min_prev / (rg.min_prev + 500.0)).fillna(0.0)
    rg["exp_mp"] = rg.mpg_last10.fillna(rg.mpg_to_date).fillna(rg.mpg_prev).fillna(0.0)
    rg["rotation"] = rg.mpg_to_date.fillna(rg.mpg_prev).fillna(0.0) >= ROTATION_MPG
    rg["streak_bin"] = pd.cut(rg.streak_before, STREAK_BINS, labels=STREAK_LABELS).astype(str)
    rg["y_played"] = played
    return rg


def attach_reports(pgf: pd.DataFrame, availability: pd.DataFrame, team_status: pd.DataFrame) -> pd.DataFrame:
    """Pre-game report status per row: a status, 'not_listed' (usable snapshot, player not on
    it) or 'no_report' (no usable snapshot for that team-game)."""
    a = availability[availability.player_id.notna()][["game_id", "player_id", "injury_status", "reason"]]
    a = a.drop_duplicates(["game_id", "player_id"])
    x = pgf.merge(a, on=["game_id", "player_id"], how="left")
    ts = team_status[["game_id", "team_id", "snapshot_usable"]].assign(team_id=lambda t: t.team_id.astype(int))
    x = x.assign(team_id=x.team_id.astype(int)).merge(ts, on=["game_id", "team_id"], how="left")
    usable = x.snapshot_usable.astype("boolean").fillna(False).astype(bool)
    x["status"] = np.where(x.injury_status.notna(), x.injury_status, np.where(usable, "not_listed", "no_report"))
    x["reason_cat"] = [reason_category(r) if s not in ("not_listed", "no_report") else s
                       for r, s in zip(x.reason, x.status)]
    return x


# ------------------------------------------------------------------ availability model
AV_FEATURES = ["status", "rotation", "reason_cat", "streak_bin"]


def _design(x: pd.DataFrame, levels: dict | None = None):
    cols = {
        "status_rot": x.status + "|" + x.rotation.map({True: "rot", False: "bench"}),
        "reason_cat": x.reason_cat,
        "streak_bin": x.streak_bin,
        "first_game": (x.gp_before == 0).map({True: "first", False: "later"}),
    }
    if levels is None:
        levels = {k: sorted(v.unique()) for k, v in cols.items()}
    mats = [np.ones((len(x), 1))]
    for k, v in cols.items():
        mats.append(np.column_stack([(v == lv).to_numpy(float) for lv in levels[k][1:]]) if len(levels[k]) > 1
                    else np.zeros((len(x), 0)))
    return np.hstack(mats), levels


def fit_availability(x: pd.DataFrame, l2: float = 1.0) -> dict:
    """L2-regularised logistic regression (Newton) for P(played)."""
    X, levels = _design(x)
    y = x.y_played.to_numpy(float)
    b = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (p - y) + pen * b
        H = (X.T * (p * (1 - p))) @ X + np.diag(pen)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"beta": b, "levels": levels}


def predict_availability(model: dict, x: pd.DataFrame) -> np.ndarray:
    X, _ = _design(x, model["levels"])
    return 1 / (1 + np.exp(-np.clip(X @ model["beta"], -30, 30)))


def status_table_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Phase 2-style baseline: P(played | status, rotation) from training frequencies."""
    key_tr = train.status + "|" + train.rotation.astype(str)
    tab = train.y_played.groupby(key_tr).mean()
    key_te = test.status + "|" + test.rotation.astype(str)
    return key_te.map(tab).fillna(train.y_played.mean()).clip(1e-4, 1 - 1e-4).to_numpy(float)


# ------------------------------------------------------------------ lineup contribution
def lineup_features(x: pd.DataFrame, q: np.ndarray, games: pd.DataFrame, tau_days: float) -> pd.DataFrame:
    """Per team-game, for value (V, relative to an average player) and minutes (f = E[MP]/48,
    whose coefficient is the drop to replacement level), each as
      *_today   expected missing today (1 - q)
      *_recent  time-decayed mean of what was actually missing in earlier games this season
      d*        today - recent (not yet absorbed by the dynamic strength)
      *_actual  actually missing this game (oracle, post-game; evaluation only)"""
    x = x.assign(q=q, f=x.exp_mp / 48.0)
    x["exp_v"] = (1 - x.q) * x.f * x.V
    x["exp_f"] = (1 - x.q) * x.f
    x["act_v"] = (1 - x.y_played) * x.f * x.V
    x["act_f"] = (1 - x.y_played) * x.f
    t = x.groupby(["season", "game_id", "team_id"]).agg(
        Mv_today=("exp_v", "sum"), Mf_today=("exp_f", "sum"), Mv_actual=("act_v", "sum"),
        Mf_actual=("act_f", "sum"), date=("game_date", "first")).reset_index()
    t = t.sort_values(["season", "team_id", "date"]).reset_index(drop=True)
    for kind in ("v", "f"):
        rec = np.full(len(t), np.nan)
        act = t[f"M{kind}_actual"].to_numpy()
        for _, idx in t.groupby(["season", "team_id"]).groups.items():
            idx = np.asarray(sorted(idx, key=lambda i: t.date.iloc[i]))
            d = pd.to_datetime(t.date.iloc[idx]).to_numpy()
            a = act[idx]
            for j in range(1, len(idx)):
                age = (d[j] - d[:j]).astype("timedelta64[D]").astype(float)
                w = np.exp(-age / tau_days)
                rec[idx[j]] = float(np.sum(w * a[:j]) / np.sum(w))
        t[f"M{kind}_recent"] = rec
        t[f"dM{kind}"] = t[f"M{kind}_today"] - t[f"M{kind}_recent"].fillna(0.0)
    return t


# ------------------------------------------------------------------ return-time tables
def return_table(pgf: pd.DataFrame, max_streak: int = 30) -> pd.DataFrame:
    """Rotation players with >=5 games played: P(plays the next team game | already missed n
    in a row). Uses actual participation only (all seasons)."""
    r = pgf[pgf.rotation & (pgf.gp_before >= 5) & (pgf.streak_before >= 1) & (pgf.streak_before <= max_streak)]
    out = r.groupby("streak_before").y_played.agg(p_return_next="mean", n="size").reset_index()
    # P(back within k more games | missed n) from the hazard sequence
    hz = out.set_index("streak_before").p_return_next
    for k in (5, 10, 20):
        out[f"p_back_within_{k}"] = [1 - np.prod([1 - hz.get(n + i, hz.iloc[-1]) for i in range(k)]) for n in out.streak_before]
    return out


def return_by_reason(x: pd.DataFrame) -> pd.DataFrame:
    r = x[x.rotation & (x.gp_before >= 5) & x.status.isin(["Out", "Doubtful", "Questionable"])]
    return r.groupby(["reason_cat", "status"]).y_played.agg(p_played="mean", n="size").reset_index()


# ------------------------------------------------------------------ evaluation
def _ols(tr: pd.DataFrame, cols: list[str]):
    X = np.column_stack([tr[c].fillna(0).to_numpy(float) for c in cols])
    b, *_ = np.linalg.lstsq(X, tr.y.to_numpy(float), rcond=None)
    resid = tr.y.to_numpy(float) - X @ b
    sigma = float(np.std(resid))
    cov = sigma ** 2 * np.linalg.pinv(X.T @ X)  # pinv: an all-zero column (no reports) gets se 0
    return b, sigma, np.sqrt(np.diag(cov))


def evaluate(gx: pd.DataFrame, test_seasons: list[str]) -> tuple[pd.DataFrame, dict]:
    from .game_model import GROUPS, _fit_catboost
    specs = {
        "dyn": ["dyn_diff", "home_flag"],
        "dyn+phase2_injury": ["dyn_diff", "nout_d", "mmin_d", "home_flag"],
        "dyn+M_today": ["dyn_diff", "Mv_today_d", "Mf_today_d", "home_flag"],
        "dyn+dM": ["dyn_diff", "dMv_d", "dMf_d", "home_flag"],
        "dyn+M_today+M_recent": ["dyn_diff", "Mv_today_d", "Mf_today_d", "Mv_recent_d", "Mf_recent_d", "home_flag"],
        "oracle_actual": ["dyn_diff", "Mv_actual_d", "Mf_actual_d", "home_flag"],
    }
    base_cb = GROUPS["strength"] + GROUPS["rest"] + GROUPS["travel"] + GROUPS["injury"] + ["home_flag"]
    p3_cols = ["Mv_today_d", "Mf_today_d", "dMv_d", "dMf_d", "Mv_recent_d", "Mf_recent_d"]
    rows, coefs = [], {}
    for test in test_seasons:
        tr = gx[(gx.season.map(season_start_year) < season_start_year(test)) & gx.report_season]
        te = gx[gx.season == test]
        res = {"season": test, "n_games": len(te), "train_games": len(tr)}
        for name, cols in specs.items():
            b, sg, se = _ols(tr, cols)
            pred = np.column_stack([te[c].fillna(0).to_numpy(float) for c in cols]) @ b
            res[f"{name}_log_loss"] = M.log_loss(te.home_win.astype(float), norm.cdf(pred / sg))
            res[f"{name}_mae"] = M.mae(te.y, pred)
            coefs.setdefault(name, {})[test] = {c: (float(v), float(s_)) for c, v, s_ in zip(cols, b, se)}
        # CatBoost: Phase 2 full feature set vs + Phase 3 lineup features (all training seasons)
        tr_all = gx[gx.season.map(season_start_year) < season_start_year(test)]
        tr_all = tr_all[tr_all.season.map(season_start_year) >= 2013]
        valid = tr_all[tr_all.season == tr_all.season.max()]
        fit_tr = tr_all[tr_all.season != tr_all.season.max()]
        for name, cols in {"cb_phase2": base_cb, "cb_phase3": base_cb + p3_cols}.items():
            mdl, sg = _fit_catboost(fit_tr, valid, cols)
            pred = mdl.predict(te[cols])
            res[f"{name}_log_loss"] = M.log_loss(te.home_win.astype(float), norm.cdf(pred / sg))
            res[f"{name}_mae"] = M.mae(te.y, pred)
        rows.append(res)
        log.info("phase3 %s: dyn %.4f p2inj %.4f dM %.4f cb2 %.4f cb3 %.4f", test, res["dyn_log_loss"],
                 res["dyn+phase2_injury_log_loss"], res["dyn+dM_log_loss"], res["cb_phase2_log_loss"],
                 res["cb_phase3_log_loss"])
    return pd.DataFrame(rows), coefs


def run(paths: Paths, cfg: dict) -> dict:
    Pd = paths.processed
    games = pd.read_parquet(Pd / "games.parquet")
    pg = pd.read_parquet(Pd / "player_games.parquet")
    pv = pd.read_parquet(Pd / "player_values.parquet")
    avail = pd.read_parquet(Pd / "player_availability.parquet")
    ts = pd.read_parquet(Pd / "injury_team_status.parquet")
    gx = pd.read_parquet(Pd / "game_model_features.parquet")
    import json
    tau = json.loads((paths.reports / "phase2_params.json").read_text())["tuned"]["dynamic"]["tau"]
    # seasons whose pre-game snapshots actually exist in the build (>50% of team-games usable);
    # configured seasons still being downloaded are excluded until the next build
    share = ts.groupby("season").snapshot_usable.mean()
    report_seasons = [s for s in cfg["data_sources"]["injury_reports"]["regular_season_seasons"]
                      if share.get(s, 0.0) > 0.5]

    pgf = player_game_frame(games, pg, pv)
    x = attach_reports(pgf, avail, ts)

    # availability model: rolling by season over report seasons
    av_rows, q = [], np.full(len(x), np.nan)
    rs = sorted(report_seasons, key=season_start_year)
    for i, s in enumerate(rs):
        te_m = (x.season == s).to_numpy()
        if i == 0:
            continue
        tr = x[x.season.isin(rs[:i]) & (x.status != "no_report")]
        mdl = fit_availability(tr)
        te = x[te_m]
        p_m = predict_availability(mdl, te)
        p_b = status_table_baseline(tr, te)
        q[te_m] = p_m
        rot = te.rotation.to_numpy()
        yv = te.y_played.to_numpy(float)
        av_rows.append({"season": s, "rows": int(te_m.sum()),
                        "model_log_loss": M.log_loss(yv, p_m), "status_table_log_loss": M.log_loss(yv, p_b),
                        "model_log_loss_rotation": M.log_loss(yv[rot], p_m[rot]),
                        "status_table_log_loss_rotation": M.log_loss(yv[rot], p_b[rot])})
    av = pd.DataFrame(av_rows)
    # seasons without reports (or the first report season): availability from history only
    hist_only = np.isnan(q)
    if hist_only.any():
        # report-free model (role, missed streak, first game only). Fit on report seasons and
        # applied to seasons outside the evaluation window, so no test season is affected.
        tr = x[x.season.isin(rs)].assign(status="no_report", reason_cat="no_report")
        mdl0 = fit_availability(tr)
        q[hist_only] = predict_availability(mdl0, x[hist_only].assign(status="no_report", reason_cat="no_report"))

    lf = lineup_features(x, q, games, tau)
    for c in ["Mv_today", "Mf_today", "Mv_actual", "Mf_actual", "Mv_recent", "Mf_recent", "dMv", "dMf"]:
        side = lf[["game_id", "team_id", c]]
        for s_ in ("home", "away"):
            gx = gx.merge(side.rename(columns={"team_id": f"{s_}_team_id", c: f"{c}_{s_}"}), on=["game_id", f"{s_}_team_id"], how="left")
        gx[f"{c}_d"] = gx[f"{c}_home"] - gx[f"{c}_away"]
    gx["nout_d"] = gx.nout_home - gx.nout_away
    gx["mmin_d"] = gx.mmin_home - gx.mmin_away
    first_eval = rs[2] if len(rs) > 2 else rs[-1]
    gx["report_season"] = gx.season.isin(rs[1:])
    tests = [s for s in rs if season_start_year(s) >= season_start_year(first_eval)]
    ev, coefs = evaluate(gx, tests)

    surv = return_table(pgf)
    reasons = return_by_reason(x[x.season.isin(rs)])
    info = {"share_value_from_players_without_prev_season": float(
        (x.exp_mp * (~x.has_prev)).sum() / x.exp_mp.sum()), "tau_days": tau, "test_seasons": tests,
        "availability_first_test": rs[1], "report_seasons_used": rs}
    return {"availability": av, "evaluation": ev, "coefs": coefs, "survival": surv, "reasons": reasons,
            "info": info, "game_features": gx, "lineup": lf}
