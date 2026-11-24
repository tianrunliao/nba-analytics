"""Rolling playoff evaluation and pre-playoff strength rankings.

    python -m nba_sos.pipeline playoff-rank

For every target season Y (from the 5th historical season on), parameters are chosen and
fit on earlier postseasons only; Y's own playoff results are used solely for evaluation
and for the post-hoc "injury-adjusted playoff performance" column.

``full`` is the forecast (pre-playoff inputs only). ``full_injury_adjusted`` adds d_h_inplay,
built from the players who actually logged no minutes in each playoff game: a post-hoc
reference, not a forecast.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import Paths, season_start_year
from ..evaluation import metrics as M
from ..ingestion import nba_stats
from ..ingestion.raw_store import RawStore
from . import players as P
from . import playoff_model as PM
from . import team_strength as T

log = logging.getLogger(__name__)
MIN_TRAIN_POSTSEASONS = 4


def _load(paths: Paths) -> dict:
    r = lambda n: pd.read_parquet(paths.processed / f"{n}.parquet")  # noqa: E731
    return {
        n: r(n)
        for n in [
            "games",
            "team_game_features",
            "player_games",
            "player_availability",
            "playoff_games",
            "playoff_player_games",
            "playoff_start_availability",
            "playoff_start_team_status",
        ]
    }


def _ols(rows: pd.DataFrame, cols: list[str]) -> dict:
    X, y = rows[cols].to_numpy(float), rows.y.to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return {"coef": dict(zip(cols, b)), "sigma": float(np.sqrt(res @ res / (len(y) - len(cols))))}


def _miss_seasons(season: str, rs_report_seasons: list[str]) -> tuple[list[str], bool]:
    """Seasons whose RS reports estimate P(miss | status): those up to Y. Before the first
    RS report season the first one is used (look-ahead, affects training seasons only)."""
    s = [x for x in rs_report_seasons if season_start_year(x) <= season_start_year(season)]
    return (s, False) if s else (rs_report_seasons[:1], True)


def _streak_rule(
    rg: pd.DataFrame, values: pd.DataFrame, streaks: dict, d: dict, hist: list[str], missing_statuses: list[str]
) -> dict:
    """Threshold of the no-report proxy (:func:`playoff_model.choose_streak_threshold`),
    scored on rotation players of teams with a usable pre-playoff report: 'out' = listed
    with a missing status. Uses pre-playoff information only (no playoff result)."""
    ts, st = d["playoff_start_team_status"], d["playoff_start_availability"]
    parts = []
    for s in hist:
        usable = set(ts[(ts.season == s) & ts.snapshot_usable].team_id.astype(int))
        if not usable:
            continue
        r = PM.season_end_roster(rg, values, s)
        r = r[(r.mpg >= PM.ROTATION_MPG) & r.team_id.astype(int).isin(usable)][["player_id", "team_id"]]
        r = r.assign(team_id=r.team_id.astype(int)).merge(streaks[s], on=["player_id", "team_id"])
        out = st[(st.season == s) & st.injury_status.isin(missing_statuses) & st.player_id.notna()]
        keys = set(zip(out.player_id.astype(int), out.team_id.astype(int)))
        parts.append(r.assign(out=[(p, t) in keys for p, t in zip(r.player_id.astype(int), r.team_id)], season=s))
    rule = PM.choose_streak_threshold(pd.concat(parts, ignore_index=True))
    rule["seasons"] = sorted({p.season.iloc[0] for p in parts if len(p)})
    return rule


def standings(paths: Paths, season: str) -> pd.DataFrame:
    s = nba_stats.load(RawStore(paths.raw), "standings", season)
    return pd.DataFrame(
        {
            "team_id": s.TeamID.astype(int),
            "conference": s.Conference,
            "seed": pd.to_numeric(s.PlayoffRank, errors="coerce"),
            "wins": s.WINS.astype(int),
            "losses": s.LOSSES.astype(int),
        }
    )


def season_blocks(
    paths: Paths,
    cfg: dict,
    d: dict,
    pv: pd.DataFrame | None = None,
    value_source: str = "onoff",
    heteroscedastic: bool = False,
) -> dict:
    """``value_source``: 'onoff' (box-score on/off, shrunk) or 'lpm' (lineup RAPM with box
    prior, from ``pv``). ``pv`` also enables the star / depth team columns."""
    hist = cfg["data_sources"]["seasons"]["historical"]
    # only seasons whose regular-season reports are actually parsed in this build
    parsed = set(d["player_availability"].season.unique())
    rs_report_seasons = [s for s in cfg["data_sources"]["injury_reports"]["regular_season_seasons"] if s in parsed]
    games, pg = d["games"], d["player_games"]
    rg = P.roster_games(pg, games)
    values = P.player_values(rg)
    if value_source == "lpm":
        assert pv is not None, "value_source='lpm' needs pv"
        values = values.drop(columns="value").merge(
            pv[["season", "player_id", "lpm"]].rename(columns={"lpm": "value"}), on=["season", "player_id"], how="left"
        )
        values["value"] = values.value.fillna(0.0)
    hi = P.health_index(rg, values)
    gf = T.game_frame(games, d["team_game_features"], hi)
    names = pg.drop_duplicates("player_id").set_index("player_id").player_name
    streaks = {s: PM.rs_end_absence_streak(rg, s) for s in hist}
    streak_rule = _streak_rule(rg, values, streaks, d, hist, cfg["features"]["injury"]["missing_statuses"])
    out = {}
    for s in hist:
        fit = T.fit_pooled(
            gf[gf.season.map(season_start_year) <= season_start_year(s)], heteroscedastic=heteroscedastic
        )
        gs = gf[gf.season == s]
        recents = {tau: T.fit_recent(gs, fit, tau) for tau in PM.TAU_GRID}
        roster = PM.season_end_roster(rg, values, s)
        pgames = d["playoff_games"][d["playoff_games"].season == s]
        ppg = d["playoff_player_games"][d["playoff_player_games"].season == s]
        play_teams = set(pd.concat([pgames.home_team_id, pgames.away_team_id]).astype(int))
        roster = roster.assign(team_id=roster.team_id.astype(int))
        ts = d["playoff_start_team_status"]
        usable = set(ts[(ts.season == s) & ts.snapshot_usable].team_id.astype(int))
        miss_seasons, lookahead = _miss_seasons(s, rs_report_seasons)
        mt = PM.status_miss_table(d["player_availability"], pg, miss_seasons)
        st = d["playoff_start_availability"]
        st = st[(st.season == s) & st.player_id.notna() & st.team_id.notna()].assign(
            team_id=lambda x: x.team_id.astype(int)
        )
        r_rep, r_prx = (
            roster[roster.team_id.isin(usable & play_teams)],
            roster[roster.team_id.isin(play_teams - usable)],
        )
        parts, details = [], []
        if len(r_rep):
            h, det = PM.health_start(r_rep, st, None, mt, fit["coefs"])
            parts.append(h)
            details.append(det)
        if len(r_prx):
            h, det = PM.health_start(r_prx, None, streaks[s], None, fit["coefs"], streak_k=streak_rule["k"])
            parts.append(h)
            details.append(det)
        hs = pd.concat(parts, ignore_index=True)
        detail = pd.concat(details, ignore_index=True)
        inplay, ip_detail = PM.inplay_health(ppg, roster, detail, pgames, fit["coefs"])
        teams = {tau: PM.team_table(fit["theta"][fit["theta"].season == s], recents[tau], hs, s) for tau in PM.TAU_GRID}
        team_cols = list(PM.TEAM_COLS)
        if pv is not None:
            from .player_report import team_star_depth

            r = (
                roster[["player_id", "team_id", "mpg"]]
                .merge(pv[pv.season == s][["player_id", "lpm"]], on="player_id", how="left")
                .fillna({"lpm": 0.0})
            )
            sd = team_star_depth(r)
            teams = {tau: t.merge(sd, on="team_id", how="left") for tau, t in teams.items()}
            team_cols += ["star", "depth"]
        rows = {tau: PM.playoff_rows(pgames, teams[tau], inplay, team_cols) for tau in PM.TAU_GRID}
        raw = games[games.season == s]
        raw_net = pd.concat(
            [
                raw[["home_team_id", "home_score", "away_score", "possessions"]].set_axis(
                    ["team_id", "pf", "pa", "poss"], axis=1
                ),
                raw[["away_team_id", "away_score", "home_score", "possessions"]].set_axis(
                    ["team_id", "pf", "pa", "poss"], axis=1
                ),
            ]
        )
        raw_net = (
            raw_net.astype(float)
            .groupby("team_id")
            .apply(lambda x: (x.pf.sum() - x.pa.sum()) / x.poss.sum() * 100, include_groups=False)
            .rename("raw_net")
        )
        out[s] = {
            "fit": fit,
            "teams": teams,
            "rows": rows,
            "team_cols": team_cols,
            "season_sigma": fit["season_sigma"][s],
            "detail": detail,
            "ip_detail": ip_detail,
            "inplay_table": inplay,
            "pgames": pgames,
            "raw_net": raw_net,
            "miss_table": mt,
            "miss_lookahead": lookahead,
            "streak_rule": streak_rule,
            "n_proxy_teams": len(play_teams - usable),
            "names": names,
            "standings": standings(paths, s),
        }
        log.info(
            "%s: theta fit on %d games, %d playoff games",
            s,
            len(gf[gf.season.map(season_start_year) <= season_start_year(s)]),
            len(pgames),
        )
    return out


def _baseline_rows(rows: pd.DataFrame, raw_net: pd.Series, st: pd.DataFrame) -> pd.DataFrame:
    wp = st.set_index("team_id").pipe(lambda x: x.wins / (x.wins + x.losses))
    r = rows.copy()
    r["d_raw_net"] = r.home_team_id.map(raw_net) - r.away_team_id.map(raw_net)
    r["d_winpct"] = r.home_team_id.map(wp) - r.away_team_id.map(wp)
    return r


def _series_eval(pgames: pd.DataFrame, rating: pd.Series, hca: float, sigma: float) -> pd.DataFrame:
    from ..preprocessing.playoffs import series_table

    g = pgames.assign(home_team_id=pgames.home_team_id.astype(int), away_team_id=pgames.away_team_id.astype(int))
    sr = series_table(g)
    g1 = g.sort_values("game_id").assign(key=lambda x: x.game_id.str[:9]).drop_duplicates("key")
    home1 = dict(zip(g1.key, g1.home_team_id))
    sr["key"] = [f"004{s[2:4]}00{r}{n}" for s, r, n in zip(sr.season, sr["round"], sr.series_no)]
    sr["lo_home"] = sr.key.map(home1) == sr.team_lo
    neutral = bool(g.is_neutral.all())
    sr["p_lo"] = [
        PM.series_win_prob(rating[a], rating[b], h, 0.0 if neutral else hca, sigma)
        for a, b, h in zip(sr.team_lo, sr.team_hi, sr.lo_home)
    ]
    sr["lo_won"] = (sr.winner == sr.team_lo).astype(int)
    return sr


def scale_blocks(blocks: dict) -> dict:
    """Per-season noise standardisation: divide every per-100 quantity of season s by that
    season's regular-season game-noise sd (known before its playoffs)."""
    out = {}
    for s, b in blocks.items():
        sg = b["season_sigma"]
        nb = dict(b)
        nb["rows"] = {}
        for tau, r in b["rows"].items():
            r = r.copy()
            dcols = [c for c in r.columns if c.startswith("d_")] + ["y"]
            r[dcols] = r[dcols] / sg
            nb["rows"][tau] = r
        nb["teams"] = {
            tau: t.assign(**{c: t[c] / sg for c in b["team_cols"] if c in t}) for tau, t in b["teams"].items()
        }
        nb["raw_net"] = b["raw_net"]
        out[s] = nb
    return out


def evaluate_and_rank(blocks: dict, hist: list[str], feats: list[str] | None = None) -> tuple[pd.DataFrame, dict, dict]:
    feats = feats or PM.FEATS
    metrics, rankings, params = [], {}, {}
    for i, Y in enumerate(hist):
        if i < MIN_TRAIN_POSTSEASONS:
            continue
        train = hist[:i]
        best = None
        for tau in PM.TAU_GRID:
            pm = PM.fit_playoff(pd.concat([blocks[s]["rows"][tau] for s in train], ignore_index=True), feats)
            if best is None or pm["loglik"] > best[1]["loglik"]:
                best = (tau, pm)
        assert best is not None
        tau, pm = best
        b = blocks[Y]
        test = b["rows"][tau]
        # baselines, all fit on the same training postseasons
        tr = pd.concat(
            [_baseline_rows(blocks[s]["rows"][tau], blocks[s]["raw_net"], blocks[s]["standings"]) for s in train]
        )
        te = _baseline_rows(test, b["raw_net"], b["standings"])
        base = {
            name: _ols(tr, cols)
            for name, cols in {
                "theta_only": ["d_theta_full", "home_flag"],
                "raw_net": ["d_raw_net", "home_flag"],
                "winpct": ["d_winpct", "home_flag"],
            }.items()
        }
        y_win = test.home_win.astype(float).to_numpy()
        res = {"season": Y, "train_postseasons": len(train), "tau_days": tau, "n_games": len(test)}
        preds = {
            "full": (PM.predict_rows(test, pm), pm["sigma"]),
            "full_injury_adjusted": (PM.predict_rows(test, pm, with_inplay=True), pm["sigma"]),
        }
        for name, bm in base.items():
            preds[name] = (sum(bm["coef"][c] * te[c] for c in bm["coef"]).to_numpy(float), bm["sigma"])
        for name, (m, sg) in preds.items():
            res[f"{name}_log_loss"] = M.log_loss(y_win, norm.cdf(m / sg))
            res[f"{name}_mae"] = M.mae(test.y, m)
        # series
        teams = b["teams"][tau].assign(team_id=lambda x: x.team_id.astype(int)).set_index("team_id")
        teams["rating"] = PM.rating(teams, pm)
        sr = _series_eval(b["pgames"], teams.rating, pm["coef"]["home_flag"], pm["sigma"])
        res["series"] = len(sr)
        res["full_series_acc"] = float(((sr.p_lo > 0.5) == (sr.lo_won == 1)).mean())
        res["full_series_brier"] = M.brier(sr.lo_won, sr.p_lo)
        for name in ["raw_net", "winpct", "theta_only"]:
            bm = base[name]
            col = {
                "raw_net": b["raw_net"],
                "theta_only": teams.theta_full,
                "winpct": b["standings"].set_index("team_id").pipe(lambda x: x.wins / (x.wins + x.losses)),
            }[name]
            k = bm["coef"][[c for c in bm["coef"] if c != "home_flag"][0]]
            srb = _series_eval(b["pgames"], col * k, bm["coef"]["home_flag"], bm["sigma"])
            res[f"{name}_series_acc"] = float(((srb.p_lo > 0.5) == (srb.lo_won == 1)).mean())
            res[f"{name}_series_brier"] = M.brier(srb.lo_won, srb.p_lo)
        metrics.append(res)
        params[Y] = {
            "tau_days": tau,
            **{f"c_{k}": v for k, v in pm["coef"].items()},
            **{f"se_{k}": v for k, v in pm["se"].items()},
            "sigma": pm["sigma"],
            "n_train_games": pm["n_games"],
            "rs_coefs": b["fit"]["coefs"],
            "miss_table": b["miss_table"].to_dict("records"),
            "miss_table_lookahead": b["miss_lookahead"],
            "n_proxy_teams": b["n_proxy_teams"],
            "rs_end_streak_rule": b["streak_rule"],
        }
        rankings[Y] = ranking_table(b, teams, pm, sr, tau)
    return pd.DataFrame(metrics), rankings, params


def ranking_table(b: dict, teams: pd.DataFrame, pm: dict, sr: pd.DataFrame, tau: float) -> pd.DataFrame:
    c = pm["coef"]
    t = teams.copy()
    st = b["standings"].set_index("team_id")
    t = t.join(st)
    t["raw_net"] = b["raw_net"]
    t["momentum_effect"] = c["d_momentum"] * t.momentum
    t["health_effect"] = c["d_h_start"] * t.h_start
    # round reached
    rr = (
        pd.concat(
            [
                sr[["team_lo", "round"]].rename(columns={"team_lo": "team_id"}),
                sr[["team_hi", "round"]].rename(columns={"team_hi": "team_id"}),
            ]
        )
        .groupby("team_id")["round"]
        .max()
    )
    champ = sr[sr["round"] == 4].winner
    t["round_reached"] = rr
    t.loc[t.index.isin(champ), "round_reached"] = 5
    # key absences at start and during the playoffs
    names = b["names"]
    det = b["detail"]
    # display only: players with a real chance of missing (the rating already weights every
    # listed player by p_miss, including 'Available' ones with a few percent)
    det = det[det.p_miss >= 0.3].assign(w=lambda x: x.p_miss * x.f * (1 + x.value.clip(lower=0)))
    det = det.sort_values("w", ascending=False)
    det = det[det.mpg >= 15]
    absent = {
        tid: ", ".join(f"{names.get(p, p)}({st})" for p, st in zip(x.player_id.head(3), x.injury_status.head(3)))
        for tid, x in det.groupby("team_id")
    }
    t["absent_at_start"] = t.index.map(absent)
    ip = b["ip_detail"]
    if len(ip):
        miss = ip.groupby(["team_id", "player_id"]).size().rename("g").reset_index()
        miss = miss.sort_values("g", ascending=False)
        missed = {
            tid: ", ".join(f"{names.get(p, p)}×{g}" for p, g in zip(x.player_id.head(3), x.g.head(3)))
            for tid, x in miss.groupby("team_id")
        }
        t["missed_in_playoffs"] = t.index.map(missed)
    # post-hoc, injury-adjusted playoff performance (uses Y's playoff results; evaluation only)
    rows = b["rows"][tau]
    perf = []
    for side, opp, sign in (("home", "away", 1), ("away", "home", -1)):
        x = rows.assign(team_id=rows[f"{side}_team_id"], opp_id=rows[f"{opp}_team_id"])
        own = (
            sign * (x.y - c["home_flag"] * x.home_flag)
            + x.opp_id.map(t.rating)
            - c["d_h_inplay"] * (x[f"{side}_h_inplay"] - x[f"{opp}_h_inplay"])
        )
        perf.append(pd.DataFrame({"team_id": x.team_id, "perf": own}))
    perf = pd.concat(perf).groupby("team_id").perf.agg(["mean", "size"])
    t["playoff_perf_injury_adj"] = perf["mean"]
    t["playoff_games"] = perf["size"]
    t = t[t.index.isin(perf.index)].sort_values("rating", ascending=False)
    t["rank"] = np.arange(1, len(t) + 1)
    return t.reset_index()


def run(paths: Paths, cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    d = _load(paths)
    hist = cfg["data_sources"]["seasons"]["historical"]
    blocks = season_blocks(paths, cfg, d)
    return evaluate_and_rank(blocks, hist)
