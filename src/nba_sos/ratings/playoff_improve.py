"""Improving the playoff model with past postseasons: candidate feature groups chosen by
nested, time-ordered validation.

For every test postseason Y:
  1. training postseasons = all earlier ones;
  2. forward selection over candidate groups, scored by leave-one-postseason-out CV log loss
     within the training postseasons only (a group must cut CV log loss by MIN_GAIN);
  3. fit on all training postseasons, predict Y.
Y's results are never used for any choice. d_h_inplay (players who actually sat out a
playoff game) is fitted as a control and set to 0 in every prediction (CV and test). Fixed variants (base + each single group) are
also reported to show each group's marginal out-of-sample value.

    python -m nba_sos.pipeline playoff-improve
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import Paths
from ..evaluation import metrics as M
from ..ingestion.raw_store import RawStore
from . import playoff_features as F
from . import playoff_report as R

log = logging.getLogger(__name__)
TAU = 40.0
MIN_GAIN = 0.0005
BASE = ["d_theta_full", "d_momentum", "d_h_start", "d_h_inplay", "home_flag"]
FORECAST_ZERO = {"d_h_inplay"}
GROUPS = {
    "elo_end": ["d_elo_end"],
    "playoff_shift": ["d_playoff_shift"],
    "po_form": ["d_po_form"],
    "style": [f"d_{c}" for c in F.STYLE_COLS],
    "experience_age": ["d_experience", "d_age"],
    "star_depth": ["d_star", "d_depth"],
}


def _fit(rows: pd.DataFrame, cols: list[str], ridge: float = 0.0) -> tuple[np.ndarray, float]:
    X = rows[cols].fillna(0.0).to_numpy(float)
    y = rows.y.to_numpy(float)
    pen = np.full(len(cols), ridge)
    pen[cols.index("home_flag")] = 0.0
    b = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    return b, float(np.std(y - X @ b))


def _logloss(rows, cols, b, sg) -> float:
    """Forecast log loss: d_h_inplay (actual absences in that playoff game) is a training
    control only and is zeroed when predicting, as in playoff_report's ``full``."""
    X = rows[cols].fillna(0.0).assign(**{c: 0.0 for c in cols if c in FORECAST_ZERO})
    pred = X.to_numpy(float) @ b
    return M.log_loss(rows.home_win.astype(float).to_numpy(), norm.cdf(pred / sg))


def lopo_cv(train: dict[str, pd.DataFrame], cols: list[str]) -> float:
    tot, n = 0.0, 0
    for s, te in train.items():
        tr = pd.concat([r for k, r in train.items() if k != s], ignore_index=True)
        b, sg = _fit(tr, cols)
        tot += _logloss(te, cols, b, sg) * len(te)
        n += len(te)
    return tot / n


def forward_select(train: dict[str, pd.DataFrame]) -> tuple[list[str], list[str], float]:
    cols, chosen = list(BASE), []
    best = lopo_cv(train, cols)
    while True:
        cand = {g: lopo_cv(train, cols + c) for g, c in GROUPS.items() if g not in chosen}
        if not cand:
            break
        g, ll = min(cand.items(), key=lambda kv: kv[1])
        if best - ll < MIN_GAIN:
            break
        chosen.append(g)
        cols += GROUPS[g]
        best = ll
    return cols, chosen, best


def run(paths: Paths, cfg: dict) -> dict:
    d = R._load(paths)
    hist = cfg["data_sources"]["seasons"]["historical"]
    store = RawStore(paths.raw)
    pv = pd.read_parquet(paths.processed / "player_values.parquet")
    all_games = pd.concat([pd.read_parquet(paths.processed / "prior_season_games.parquet"), d["games"]], ignore_index=True)
    elo_params = json.loads((paths.reports / "game_model_params.json").read_text())["tuned"]["elo"]
    from ..preprocessing.players import build_player_games
    prior_po = []
    for s in ["2004-05", "2005-06", "2006-07"]:
        try:
            prior_po.append(build_player_games(store, s, "playoff_player_game_logs", "004"))
        except FileNotFoundError:
            log.warning("no playoff player logs for %s (experience history shorter)", s)
    prior_po = pd.concat(prior_po, ignore_index=True) if prior_po else d["playoff_player_games"].head(0)

    blocks = R.season_blocks(paths, cfg, d, pv=pv)
    blocks = F.augment_blocks(blocks, store, d, pv, all_games, elo_params, prior_po)
    rows = {s: blocks[s]["rows"][TAU] for s in hist}

    out_rows, selected = [], {}
    for i, Y in enumerate(hist):
        if i < R.MIN_TRAIN_POSTSEASONS:
            continue
        train = {s: rows[s] for s in hist[:i]}
        te = rows[Y]
        tr_all = pd.concat(train.values(), ignore_index=True)
        res = {"season": Y, "n_games": len(te), "later_round_games": int((te.game_id.str[7].astype(int) >= 2).sum())}
        variants = {"base": BASE, **{f"base+{g}": BASE + c for g, c in GROUPS.items()}}
        cols_sel, chosen, cv = forward_select(train)
        variants["selected"] = cols_sel
        selected[Y] = {"groups": chosen, "cv_log_loss": cv}
        later = te.game_id.str[7].astype(int) >= 2
        for name, cols in variants.items():
            b, sg = _fit(tr_all, cols)
            res[f"{name}_log_loss"] = _logloss(te, cols, b, sg)
            res[f"{name}_later_log_loss"] = _logloss(te[later], cols, b, sg)
        out_rows.append(res)
        log.info("%s selected=%s base=%.4f sel=%.4f", Y, chosen, res["base_log_loss"], res["selected_log_loss"])
    m = pd.DataFrame(out_rows)
    final_cols, final_groups, final_cv = forward_select({s: rows[s] for s in hist})
    b, sg = _fit(pd.concat([rows[s] for s in hist], ignore_index=True), final_cols)
    se = _se(pd.concat([rows[s] for s in hist], ignore_index=True), final_cols, sg)
    return {"metrics": m, "selected": selected, "blocks": blocks,
            "final": {"groups": final_groups, "cols": final_cols, "coef": dict(zip(final_cols, b)),
                      "se": dict(zip(final_cols, se)), "sigma": sg, "cv_log_loss": final_cv}}


def _se(rows, cols, sg):
    X = rows[cols].fillna(0.0).to_numpy(float)
    return np.sqrt(np.diag(sg ** 2 * np.linalg.pinv(X.T @ X)))


def pooled(m: pd.DataFrame, names: list[str], later: bool = False) -> pd.DataFrame:
    w = m.later_round_games if later else m.n_games
    suf = "_later_log_loss" if later else "_log_loss"
    return pd.DataFrame([{"variant": n, "log_loss": float((m[n + suf] * w).sum() / w.sum())} for n in names])


def series_eval(blocks: dict, hist: list[str], cols_by_variant: dict[str, list[str]]) -> pd.DataFrame:
    """Series-level out-of-sample check. Each series is predicted before its first game: team
    rating = team-level coefficients x team table (+ c_form x the team's playoff form going into
    the series, for variants that use it); best-of-7 simulation with playoff home court."""
    from ..preprocessing.playoffs import series_table
    from . import playoff_model as PM
    rows = {s: blocks[s]["rows"][TAU] for s in hist}
    out = []
    for i, Y in enumerate(hist):
        if i < R.MIN_TRAIN_POSTSEASONS:
            continue
        b = blocks[Y]
        tr = pd.concat([rows[s] for s in hist[:i]], ignore_index=True)
        pg = b["pgames"].assign(home_team_id=lambda x: x.home_team_id.astype(int),
                                away_team_id=lambda x: x.away_team_id.astype(int))
        sr = series_table(pg)
        g1 = pg.sort_values("game_id").assign(key=lambda x: x.game_id.str[:9]).drop_duplicates("key")
        form = F.po_form(b["pgames"], b["teams"][TAU].set_index("team_id").theta_full, b["fit"]["coefs"][f"hca_{Y}"])
        g1 = g1.merge(form, on="game_id")
        sr["key"] = [f"004{s[2:4]}00{r}{n}" for s, r, n in zip(sr.season, sr["round"], sr.series_no)]
        info = g1.set_index("key")
        teams = b["teams"][TAU].set_index("team_id")
        for name, cols in cols_by_variant.items():
            bb, sg = _fit(tr, cols)
            coef = dict(zip(cols, bb))
            pm = {"coef": {k: v for k, v in coef.items() if k != "d_po_form"}}
            rating = PM.rating(teams.reset_index(), pm).set_axis(teams.index)
            for r in sr.itertuples(index=False):
                g = info.loc[r.key]
                f_lo = g.home_po_form if g.home_team_id == r.team_lo else g.away_po_form
                f_hi = g.away_po_form if g.home_team_id == r.team_lo else g.home_po_form
                ra, rb = rating[r.team_lo], rating[r.team_hi]
                if "d_po_form" in coef:
                    ra, rb = ra + coef["d_po_form"] * f_lo, rb + coef["d_po_form"] * f_hi
                hca = 0.0 if bool(pg.is_neutral.all()) else coef["home_flag"]
                p = PM.series_win_prob(ra, rb, g.home_team_id == r.team_lo, hca, sg)
                out.append({"season": Y, "round": r.round, "variant": name, "p_lo": p,
                            "lo_won": int(r.winner == r.team_lo)})
    return pd.DataFrame(out)
