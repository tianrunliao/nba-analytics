"""Remaining-season simulation: historical replays (acceptance test) and the current-season projection.

    python -m nba_sos.pipeline replay     # replays 2016-17 .. 2025-26
    python -m nba_sos.pipeline project    # current season from the latest snapshot

Model selection and fitting for test season T use seasons FIRST_TRAIN .. T-1 only; inside
them, choices are made on season-grouped k-fold out-of-fold error (whole seasons held out
together, never shuffled games):
  load features       forward selection (projection.select_load)
  roster prior        blend of last season's strength and the roster value (roster_prior)
  availability terms  OOF margin error with / without them (projection.horizon_cv)
The test season itself is only replayed: at the season start, or the date by which the
median team has played 20 / 40 / 60 games, with strengths, rosters and injuries known
before that date. Remaining games are the games actually played after the cutoff; for NBA
Cup seasons, a cutoff before the quarterfinals drops the knockout and extra games and
simulates them instead.

Variants: ``full`` (all of the above), ``v1`` (the first season-simulation version: rho * last season
prior, no availability terms, load = b2b / g4 / km / |tz|), ``fixed_strength`` (full model
without strength uncertainty), ``kalman`` (the full model with Kalman-filter strengths and sds,
ratings/kalman.py, in place of the dynamic ridge strengths and the roster prior; load features
and the pre-game fit use the Kalman strength difference), ``pace`` (current win% carried forward).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..config import Paths, season_start_year
from ..ingestion import nba_stats
from ..ingestion.raw_store import RawStore
from ..ratings import dynamic as DY
from ..ratings import kalman as KF
from ..sos import availability as AV
from ..sos import cup as CUP
from ..sos import projection as PJ
from ..sos import roster_prior as RP

log = logging.getLogger(__name__)
FIRST_TRAIN = "2008-09"
TEST_SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]
CUTOFF_GAMES = [0, 20, 40, 60]
V1_LOAD = ["b2b", "g4", "km", "tz_abs"]
K_FOLDS = 5


def _dyn_params(paths: Paths) -> dict:
    return json.loads((paths.reports / "game_model_params.json").read_text())["tuned"]["dynamic"]


def _kalman_params(paths: Paths) -> dict:
    return json.loads((paths.reports / "game_model_params.json").read_text())["tuned"]["kalman"]


def load_history(paths: Paths) -> dict:
    r = lambda n: pd.read_parquet(paths.processed / f"{n}.parquet")  # noqa: E731
    games, prior = r("games"), r("prior_season_games")
    rows = PJ.game_rows(games, PJ.schedule_load(r("team_game_features")))
    x = r("game_model_features")[["game_id", "dyn_diff", "kal_diff", "y"]].merge(rows, on="game_id")
    all_games = pd.concat([prior, games], ignore_index=True)
    return {
        "games": games,
        "all_games": all_games,
        "rows": rows,
        "x": x,
        "pgf": AV.player_frame(paths),
        "final": DY.season_final_strength(all_games),
        "y": games.set_index("game_id").net_rating_margin.astype(float),
        "dyn": _dyn_params(paths),
        "kal": KF.filter_states(all_games, _kalman_params(paths)),
        "_states": {},
    }


def weekly_cutoffs(dates: pd.Series) -> list[pd.Timestamp]:
    d = pd.to_datetime(dates)
    return list(pd.date_range(d.min(), d.max(), freq="7D"))


def train_seasons(h: dict, before: str) -> list[str]:
    yr = season_start_year(before)
    return sorted(
        {s for s in h["x"].season.unique() if season_start_year(FIRST_TRAIN) <= season_start_year(s) < yr},
        key=season_start_year,
    )


# ------------------------------------------------------------------ per-cutoff inputs
def _state(h: dict, season: str, cut: pd.Timestamp):
    key = (season, cut)
    if key not in h["_states"]:
        h["_states"][key] = AV.state_at(h["pgf"][h["pgf"].season == season], cut, h["dyn"]["tau"])
    return h["_states"][key]


def prior_fn(h: dict, season: str, coef: dict | None):
    """Ridge prior at a cutoff: roster prior (coef) or None (rho * last season)."""
    if coef is None:
        return None
    pgf_s = h["pgf"][h["pgf"].season == season]
    first = h["rows"][h["rows"].season == season].date.min()

    def f(cut, teams, prev):
        return RP.prior_at(pgf_s, prev, None if cut <= first else cut, coef, teams)

    return f


def injury_fn(h: dict, season: str, tabs: tuple | None):
    if tabs is None:
        return None
    haz, pi = tabs
    ids = sorted(set(h["rows"][h["rows"].season == season].home_team_id))

    def f(cut):
        players, teams = _state(h, season, cut)
        return AV.expected_change(players, teams, haz, pi, ids)

    return f


def strengths(h: dict, season: str, cutoffs, coef, all_games=None, source: str = "dynamic") -> pd.DataFrame:
    if source == "kalman":
        # the filter is causal: the state at a cutoff uses games before it only
        return KF.strengths_asof(h["kal"], season, cutoffs)
    p = h["dyn"]
    return DY.strengths_asof(
        h["all_games"] if all_games is None else all_games,
        season,
        cutoffs,
        p["tau"],
        p["alpha"],
        p["rho"],
        prior_fn=prior_fn(h, season, coef),
    )


def horizon_table(h: dict, seasons: list[str], coef, tabs, source: str = "dynamic") -> pd.DataFrame:
    out = []
    for s in seasons:
        rs = h["rows"][h["rows"].season == s]
        st = strengths(h, s, weekly_cutoffs(rs.date), coef, source=source)
        out.append(PJ.horizon_rows(rs, st, h["y"], injury=injury_fn(h, s, tabs)).assign(season=s))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ fitting
def fit_models(h: dict, before: str, variant: str = "full") -> dict:
    """Everything the projection of season ``before`` needs, from earlier seasons only."""
    train = train_seasons(h, before)
    xt = h["x"][h["x"].season.isin(train)]
    source = "kalman" if variant == "kalman" else "dynamic"
    if source == "kalman":
        xt = xt.assign(dyn_diff=xt.kal_diff)
    if variant == "v1":
        feats, sel, coef, tabs = V1_LOAD, None, None, None
    else:
        sel = PJ.select_load(xt, K_FOLDS)
        feats = sel["features"]
        coef = (
            None
            if source == "kalman"
            else RP.fit_prior(RP.training_table(h["pgf"], h["final"], train[1:]), h["dyn"]["rho"], K_FOLDS)
        )
        pt = h["pgf"][h["pgf"].season.isin(train)]
        tabs = (AV.hazards(pt), AV.new_injury_curve(pt))
    pre = PJ.fit_pregame(xt, feats)
    hrows = horizon_table(h, train, coef, tabs, source)
    cv = PJ.horizon_cv(hrows, pre, K_FOLDS) if tabs is not None else None
    hz = PJ.fit_horizon(hrows, pre, PJ.INJ_SPECS[cv["chosen"]] if cv else [])
    return {
        "variant": variant,
        "strength_source": source,
        "train": [train[0], train[-1]],
        "pre": pre,
        "hz": hz,
        "load_selection": sel,
        "roster_prior": coef,
        "injury_tables": tabs,
        "horizon_cv": cv,
        "poss": float(h["games"][h["games"].season == train[-1]].possessions.astype(float).mean()),
    }


# ------------------------------------------------------------------ replays
def cutoff_date(rows: pd.DataFrame, n: int) -> pd.Timestamp:
    """First date by which the median team has played n games (n = 0: opening night)."""
    tg = pd.concat(
        [
            rows[["date", "home_team_id"]].set_axis(["date", "team_id"], axis=1),
            rows[["date", "away_team_id"]].set_axis(["date", "team_id"], axis=1),
        ]
    )
    if n == 0:
        return tg.date.min()
    for d in sorted(tg.date.unique()):
        gp = tg[tg.date < d].team_id.value_counts()
        if len(gp) == 30 and gp.median() >= n:
            return pd.Timestamp(d)
    raise ValueError(f"median team never reaches {n} games")


def standings_meta(store: RawStore, season: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    st = nba_stats.load(store, "standings", season)
    st = st.set_index(st.TeamID.astype(int))
    return st.Conference, st.Division, st.PlayoffRank.astype(int)


def _wins_to_date(games: pd.DataFrame, cut: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    done = games[pd.to_datetime(games.game_date) < cut]
    home_w = done.home_score > done.away_score
    w = pd.concat([done.home_team_id[home_w], done.away_team_id[~home_w]]).astype(int).value_counts()
    g = pd.concat([done.home_team_id, done.away_team_id]).astype(int).value_counts()
    return w, g


def prev_wins(h: dict, season: str) -> pd.Series:
    prev = f"{season_start_year(season) - 1}-{str(season_start_year(season))[-2:]}"
    w, _ = _wins_to_date(h["all_games"][h["all_games"].season == prev], pd.Timestamp("2100-01-01"))
    return w


def cup_split(rows: pd.DataFrame, cut: pd.Timestamp, season: str, prev_w: pd.Series, cup_cfg: dict):
    """(remaining rows without Cup knockout / extra games, CupPlan) when the knockout stage
    is still ahead of the cutoff; otherwise (rows, None)."""
    if "game_subtype" not in rows:
        return rows, None
    plan = CUP.plan(rows, season, cup_cfg, prev_w)
    if plan is None or cut >= plan.qf_date:
        return rows, None
    ids = rows.game_id.astype(str).str[-5:].astype(int)
    drop = (rows.game_subtype.astype(str) == "in-season-knockout") | (ids > 1200)
    return rows[~drop], plan


def replay(
    h: dict, store: RawStore, season: str, fits: dict, n_paths: int, cup_cfg: dict, seed: int = 0
) -> pd.DataFrame:
    rows = h["rows"][h["rows"].season == season]
    games = h["games"][h["games"].season == season]
    conf, div, prank = standings_meta(store, season)
    final_w, final_g = _wins_to_date(games, rows.date.max() + pd.Timedelta(1, unit="D"))
    pw = prev_wins(h, season)
    out = []
    for n in CUTOFF_GAMES:
        cut = cutoff_date(rows, n)
        done = games[pd.to_datetime(games.game_date) < cut]
        w_now, g_now = _wins_to_date(games, cut)
        rows_c, plan = cup_split(rows, cut, season, pw, cup_cfg)
        for name, fit in fits.items():
            st = strengths(h, season, [cut], fit["roster_prior"], source=fit["strength_source"])
            fi = injury_fn(h, season, fit["injury_tables"])
            rem = PJ.future_index(rows_c, cut)
            PJ.add_injury(rem, fi(cut) if fi is not None else None)
            sos = PJ.schedule_difficulty(PJ.team_games(rem), st, fit["pre"], fit["hz"], cut)
            hz_runs = {name: fit["hz"]}
            if name == "full":
                hz_runs["fixed_strength"] = {
                    **fit["hz"],
                    "kappa": 0.0,
                    "q_per_day": 0.0,
                    "sigma_game": fit["pre"]["sigma"],
                }
            inj_paths = None
            if fit["injury_tables"] is not None:
                players, teams = _state(h, season, cut)
                h_tab, pi_tab = fit["injury_tables"]
                inj_paths = AV.sample_paths(
                    players, teams, h_tab, pi_tab, sorted(conf.index), n_paths, np.random.default_rng(seed + 1)
                )
            for vname, hzv in hz_runs.items():
                sim = PJ.simulate(
                    rem,
                    st,
                    fit["pre"],
                    hzv,
                    cut,
                    done=done,
                    n_paths=n_paths,
                    seed=seed,
                    conference=conf,
                    division=div,
                    injuries=inj_paths,
                    cup=plan,
                    cup_done=done,
                    poss=fit["poss"],
                )
                s = PJ.summarise(sim, w_now, g_now)
                s = s.merge(sos[["team_id", "d_sos", "opp_sos", "games"]], on="team_id", how="left")
                s["variant"], s["cutoff_games"], s["cutoff"], s["season"] = vname, n, cut, season
                s["cup_simulated"] = plan is not None
                out.append(s)
        pace = pd.DataFrame({"team_id": sorted(conf.index)})
        wn, gn = w_now.reindex(pace.team_id).fillna(0).to_numpy(), g_now.reindex(pace.team_id).fillna(0).to_numpy()
        tot = final_g.reindex(pace.team_id).to_numpy(float)
        pace["exp_wins"] = wn + (tot - gn) * np.where(gn > 0, wn / np.maximum(gn, 1), 0.5)
        pace["wins_now"], pace["games_now"] = wn, gn
        pace["variant"], pace["cutoff_games"], pace["cutoff"], pace["season"] = "pace", n, cut, season
        out.append(pace)
    res = pd.concat(out, ignore_index=True)
    res["final_wins"] = res.team_id.map(final_w).fillna(0).to_numpy()
    res["final_games"] = res.team_id.map(final_g).to_numpy()
    res["top6"] = res.team_id.map(prank <= 6).astype(float)
    res["playoff_rank"] = res.team_id.map(prank)
    return res


def evaluate(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v, n), g in res.groupby(["variant", "cutoff_games"]):
        e = g.exp_wins - g.final_wins
        r = {
            "variant": v,
            "cutoff_games": n,
            "team_seasons": len(g),
            "mae": float(e.abs().mean()),
            "rmse": float(np.sqrt((e**2).mean())),
        }
        if "p05" in g and g.p05.notna().all():
            r["cover50"] = float(((g.final_wins >= g.p25) & (g.final_wins <= g.p75)).mean())
            r["cover90"] = float(((g.final_wins >= g.p05) & (g.final_wins <= g.p95)).mean())
            r["width90"] = float((g.p95 - g.p05).mean())
            r["brier_top6"] = float(((g.p_top6 - g.top6) ** 2).mean())
            r["mae_seed"] = float((g.exp_seed - g.playoff_rank).abs().mean())
        rows.append(r)
    return pd.DataFrame(rows)


def sos_check(res: pd.DataFrame) -> dict:
    """Does remaining-schedule D-SOS explain remaining wins beyond the team's own strength?
    remaining wins - G/2 = a * (expected remaining wins - G/2 + D-SOS) + b * (-D-SOS), on the
    full-model replays after >= 20 games."""
    g = res[(res.variant == "full") & (res.cutoff_games >= 20)].copy()
    rem_g = g.final_games - g.games_now
    y = (g.final_wins - g.wins_now) - rem_g / 2
    strength = (g.exp_wins - g.wins_now) - rem_g / 2 + g.d_sos
    X = np.column_stack([strength, -g.d_sos])
    b, *_ = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)
    res_ = y.to_numpy(float) - X @ b
    se = np.sqrt(np.diag(res_ @ res_ / (len(y) - 2) * np.linalg.inv(X.T @ X)))
    return {
        "coef_strength": float(b[0]),
        "coef_minus_dsos": float(b[1]),
        "se": [float(v) for v in se],
        "n": int(len(g)),
        "mae_remaining_wins_with_dsos": float(np.mean(np.abs(y - strength + g.d_sos))),
        "mae_without_dsos": float(np.mean(np.abs(y - strength))),
        "d_sos_sd": float(g.d_sos.std()),
    }


def _fit_summary(f: dict) -> dict:
    return {k: v for k, v in f.items() if k != "injury_tables"}


def run_backtest(paths: Paths, cfg: dict, n_paths: int = 2000) -> dict:
    h = load_history(paths)
    store = RawStore(paths.raw)
    cup_cfg = cfg["data_sources"].get("nba_cup", {})
    out, params = [], {}
    for T in TEST_SEASONS:
        fits = {v: fit_models(h, T, v) for v in ("full", "v1", "kalman")}
        params[T] = {k: _fit_summary(v) for k, v in fits.items()}
        log.info(
            "%s load %s prior %s hz %s cv %s",
            T,
            fits["full"]["load_selection"]["features"],
            {k: round(v, 4) for k, v in fits["full"]["roster_prior"].items()},
            {k: v for k, v in fits["full"]["hz"].items() if k in ("b0", "kappa", "q_per_day", "c_inj")},
            fits["full"]["horizon_cv"],
        )
        out.append(replay(h, store, T, fits, n_paths, cup_cfg))
    res = pd.concat(out, ignore_index=True)
    return {"replays": res, "metrics": evaluate(res), "params": params, "sos_check": sos_check(res)}


# ------------------------------------------------------------------ current season
def run_current(paths: Paths, cfg: dict, n_paths: int = 10000, seed: int = 0) -> dict:
    from ..features.pregame import back_to_back_context, schedule_density, team_game_long, travel
    from ..ingestion.venues import load_venue_config
    from ..models import availability as AM
    from ..preprocessing.games import build_schedule_games
    from ..preprocessing.players import build_player_games
    from ..sos.window_report import latest_snapshot, season_games_frame, tricodes

    season = cfg["data_sources"]["seasons"]["current"]
    last_hist = cfg["data_sources"]["seasons"]["historical"][-1]
    snap = latest_snapshot(paths)
    cur_store = RawStore(snap)
    aliases, venues = load_venue_config(paths.configs)
    g = season_games_frame(cur_store, season, aliases, venues)
    code = tricodes(cur_store, season)
    g["home_team"], g["away_team"] = g.home_team_id.map(code), g.away_team_id.map(code)
    g["season"] = season
    g["home_team_id"], g["away_team_id"] = g.home_team_id.astype(int), g.away_team_id.astype(int)
    g["possessions"] = g.possessions.astype(float)
    done = g[g.completed].copy()
    done["net_rating_margin"] = (done.home_score.astype(float) - done.away_score.astype(float)) / done.possessions * 100
    done["margin"] = done.home_score.astype(float) - done.away_score.astype(float)

    h = load_history(paths)
    fit = fit_models(h, season, "full")
    cut = (
        pd.to_datetime(done.game_date).max() + pd.Timedelta(1, unit="D")
        if len(done)
        else pd.to_datetime(g.game_date).min()
    )

    # current-season player frame (participation so far) for rosters and injuries
    pg = build_player_games(cur_store, season)
    pv = pd.read_parquet(paths.processed / "player_values.parquet")
    pgf_cur = AM.player_game_frame(done.assign(overtime_periods=done.overtime_periods.astype(float).fillna(0)), pg, pv)
    pgf_cur["game_date"] = pd.to_datetime(pgf_cur.game_date)
    h["pgf"] = pd.concat([h["pgf"], pgf_cur[h["pgf"].columns]], ignore_index=True)

    hist_cols = ["season", "game_id", "game_date", "home_team_id", "away_team_id", "is_neutral", "net_rating_margin"]
    all_games = pd.concat([h["all_games"][hist_cols], done[hist_cols]], ignore_index=True)
    t = team_game_long(g.assign(home_score=g.home_score.astype("Float64"), away_score=g.away_score.astype("Float64")))
    t = pd.concat([t, schedule_density(t, [4, 7]), travel(t, venues), back_to_back_context(t, venues)], axis=1)
    rows = PJ.game_rows(g, PJ.schedule_load(t))
    h["rows"] = pd.concat([h["rows"], rows], ignore_index=True)
    st = strengths(h, season, [cut], fit["roster_prior"], all_games=all_games)

    sched_raw = build_schedule_games(cur_store, season)
    sched_raw["date"] = pd.to_datetime(sched_raw.sched_game_date)
    sched_raw["home_team_id"] = sched_raw.sched_home_team_id.fillna(0).astype(int)
    sched_raw["away_team_id"] = sched_raw.sched_away_team_id.fillna(0).astype(int)
    sched_raw["is_neutral"] = sched_raw.source_is_neutral.astype(bool)
    plan = CUP.plan(sched_raw, season, cfg["data_sources"].get("nba_cup", {}), prev_wins(h, season))
    if plan is not None and cut >= plan.qf_date:
        plan = None
    rem = PJ.future_index(rows[rows.game_id.isin(g[~g.completed].game_id)], cut)
    players, teams_state = AV.state_at(h["pgf"][h["pgf"].season == season], cut, h["dyn"]["tau"])
    ids = sorted(st.team_id)
    h_tab, pi_tab = fit["injury_tables"]
    PJ.add_injury(rem, AV.expected_change(players, teams_state, h_tab, pi_tab, ids))
    w_now, g_now = _wins_to_date(done, cut)
    conf, div, _ = standings_meta(RawStore(paths.raw), last_hist)
    inj_paths = AV.sample_paths(players, teams_state, h_tab, pi_tab, ids, n_paths, np.random.default_rng(seed + 1))
    tg = PJ.team_games(rem)
    sim = PJ.simulate(
        rem,
        st,
        fit["pre"],
        fit["hz"],
        cut,
        done=done,
        n_paths=n_paths,
        seed=seed,
        conference=conf,
        division=div,
        injuries=inj_paths,
        cup=plan,
        cup_done=done,
        poss=float(done.possessions.mean()),
        tg_ref=tg,
    )
    summ = PJ.summarise(sim, w_now, g_now)
    sos_rem = PJ.schedule_difficulty(tg, st, fit["pre"], fit["hz"], cut)
    out = summ.merge(sos_rem, on="team_id").merge(st[["team_id", "S", "sd"]], on="team_id")
    out["team"] = out.team_id.map(code)
    out["conference"] = out.team_id.map(conf)
    out["division"] = out.team_id.map(div)
    hurt = players[players.n > 0].assign(fv=lambda d: d.f * d.V)
    out["injured_out_value"] = out.team_id.map(hurt.groupby("team_id").fv.sum()).fillna(0.0)
    out["n_injured_out"] = out.team_id.map(hurt.groupby("team_id").size()).fillna(0).astype(int)
    meta = {
        "season": season,
        "snapshot": snap.name,
        "cutoff": str(cut.date()),
        "completed_games": int(len(done)),
        "remaining_scheduled_games": int(len(rem)),
        "cup_simulated": plan is not None,
        "n_paths": n_paths,
        **_fit_summary(fit),
    }
    return {"table": out.sort_values("exp_wins", ascending=False).reset_index(drop=True), "meta": meta}
