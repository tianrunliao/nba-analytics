"""Extra pre-playoff team features and in-playoff form, for the playoff model.

All team-level inputs use the regular season of the same year (complete before its
playoffs) or earlier postseasons; ``po_form`` uses only playoff games played before the
game being predicted.

  elo_end         Elo after the last regular-season game (season-centred, /25 ~ per-100 scale)
  playoff_shift   value gained by playoff rotation shortening: sum over the top 12 players by
                  regular-season minutes of (expected playoff share - regular-season share) x LPM,
                  playoff shares by minutes rank estimated from earlier postseasons only
  style (z within season)
                  fg3a_rate, fta_rate, oreb_pct, tov_rate, pace, off_def_split
                  (offence above league minus defence above league, per 100)
  experience      minutes-weighted mean of log(1 + playoff minutes over the previous 3 postseasons)
  age             minutes-weighted mean age
  po_form         (per game) shrunk mean residual of the team's earlier games in this postseason
                  vs its pre-playoff strength: sum(resid) / (n + FORM_PRIOR_GAMES)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import season_start_year
from ..ingestion import nba_stats
from ..ingestion.raw_store import RawStore
from . import elo as E

FORM_PRIOR_GAMES = 5.0
STYLE_COLS = ["z_fg3a_rate", "z_fta_rate", "z_oreb_pct", "z_tov_rate", "z_pace", "z_off_def_split"]


def elo_end(all_games: pd.DataFrame, k: float, hca: float, carry: float) -> pd.DataFrame:
    """Elo after each season's final regular-season game."""
    g = all_games.sort_values(["tipoff_utc", "game_id"])
    pre = E.run_elo(g, k, hca, carry).set_index("game_id")
    rows = []
    for s, gs in g.groupby("season"):
        last: dict[int, float] = {}
        for r in gs.itertuples(index=False):
            rh, ra = pre.loc[r.game_id, "home_elo"], pre.loc[r.game_id, "away_elo"]
            adv = 0.0 if r.is_neutral else hca
            p = 1 / (1 + 10 ** (-(rh - ra + adv) / 400))
            mov = float(r.home_score) - float(r.away_score)
            diff_w = (rh + adv - ra) if mov > 0 else (ra - rh - adv)
            delta = k * (abs(mov) + 3) ** 0.8 / (7.5 + 0.006 * diff_w) * ((1.0 if mov > 0 else 0.0) - p)
            last[int(r.home_team_id)], last[int(r.away_team_id)] = rh + delta, ra - delta
        e = pd.Series(last)
        rows.append(pd.DataFrame({"season": s, "team_id": e.index, "elo_end": (e - e.mean()).to_numpy() / 25.0}))
    return pd.concat(rows, ignore_index=True)


def team_style(store: RawStore, season: str) -> pd.DataFrame:
    lgl = nba_stats.load(store, "league_game_log", season)
    adv = nba_stats.load(store, "team_advanced", season)
    lgl["GAME_ID"], adv["GAME_ID"] = lgl.GAME_ID.astype(str).str.zfill(10), adv.GAME_ID.astype(str).str.zfill(10)
    lgl = lgl[lgl.GAME_ID.str[:3] == "002"].merge(adv[["GAME_ID", "TEAM_ID", "POSS"]], on=["GAME_ID", "TEAM_ID"])
    opp = lgl[["GAME_ID", "TEAM_ID", "DREB", "PTS"]].rename(
        columns={"TEAM_ID": "OPP_ID", "DREB": "OPP_DREB", "PTS": "OPP_PTS"}
    )
    x = lgl.merge(opp, on="GAME_ID")
    x = x[x.TEAM_ID != x.OPP_ID]
    t = x.groupby("TEAM_ID").agg(
        FGA=("FGA", "sum"),
        FG3A=("FG3A", "sum"),
        FTA=("FTA", "sum"),
        OREB=("OREB", "sum"),
        OPP_DREB=("OPP_DREB", "sum"),
        TOV=("TOV", "sum"),
        POSS=("POSS", "sum"),
        PTS=("PTS", "sum"),
        OPP_PTS=("OPP_PTS", "sum"),
        G=("GAME_ID", "nunique"),
    )
    out = pd.DataFrame(
        {
            "team_id": t.index.astype(int),
            "fg3a_rate": t.FG3A / t.FGA,
            "fta_rate": t.FTA / t.FGA,
            "oreb_pct": t.OREB / (t.OREB + t.OPP_DREB),
            "tov_rate": t.TOV / t.POSS,
            "pace": t.POSS / t.G,
        }
    )
    ortg, drtg = t.PTS / t.POSS * 100, t.OPP_PTS / t.POSS * 100
    lg = ortg.mean()
    out["off_def_split"] = ((ortg - lg) - (lg - drtg)).to_numpy()
    for c in ["fg3a_rate", "fta_rate", "oreb_pct", "tov_rate", "pace", "off_def_split"]:
        out[f"z_{c}"] = (out[c] - out[c].mean()) / out[c].std()
    return out.reset_index(drop=True)


def playoff_share_curve(playoff_pg: pd.DataFrame, rs_pg: pd.DataFrame, seasons: list[str], top: int = 12) -> np.ndarray:
    """Mean playoff share of team minutes (x5) by regular-season minutes rank 1..top."""
    shares = []
    for s in seasons:
        rs = rs_pg[rs_pg.season == s].groupby(["team_id", "player_id"]).minutes.sum().reset_index()
        rs["rk"] = rs.groupby("team_id").minutes.rank(ascending=False, method="first")
        po = (
            playoff_pg[playoff_pg.season == s]
            .groupby(["team_id", "player_id"])
            .minutes.sum()
            .rename("po")
            .reset_index()
        )
        tot = po.groupby("team_id").po.transform("sum")
        po["po_share"] = po.po / tot * 5
        m = rs.merge(po[["team_id", "player_id", "po_share"]], on=["team_id", "player_id"])
        m = m[m.team_id.isin(po.team_id.unique())]
        full = (
            rs[rs.team_id.isin(po.team_id.unique())]
            .merge(m[["team_id", "player_id", "po_share"]], on=["team_id", "player_id"], how="left")
            .fillna({"po_share": 0.0})
        )
        shares.append(
            full[full.rk <= top].groupby("rk").po_share.mean().reindex(range(1, top + 1)).fillna(0.0).to_numpy()
        )
    return np.mean(shares, axis=0) if shares else None


def playoff_shift(roster_mpg: pd.DataFrame, curve: np.ndarray | None, top: int = 12) -> pd.DataFrame:
    """roster_mpg: team_id, player_id, rs_minutes, value. Returns team_id, playoff_shift."""
    rows = []
    for tid, g in roster_mpg.groupby("team_id"):
        g = g.sort_values("rs_minutes", ascending=False).head(top).copy()
        tot = roster_mpg[roster_mpg.team_id == tid].rs_minutes.sum()
        g["rs_share"] = g.rs_minutes / tot * 5
        po = curve[: len(g)] if curve is not None else g.rs_share.to_numpy()
        rows.append({"team_id": tid, "playoff_shift": float(np.sum((po - g.rs_share.to_numpy()) * g.value.to_numpy()))})
    return pd.DataFrame(rows)


def experience_age(roster_mpg: pd.DataFrame, prev_playoff_minutes: pd.Series, age: pd.Series) -> pd.DataFrame:
    r = roster_mpg.copy()
    r["exp"] = np.log1p(r.player_id.map(prev_playoff_minutes).fillna(0.0))
    r["age"] = r.player_id.map(age)
    out = []
    for tid, g in r.groupby("team_id"):
        w = g.rs_minutes
        out.append(
            {
                "team_id": tid,
                "experience": float(np.average(g.exp, weights=w)),
                "age": float(np.average(g.age.fillna(g.age.mean()), weights=w)),
            }
        )
    return pd.DataFrame(out)


def po_form(pgames: pd.DataFrame, theta: pd.Series, hca: float) -> pd.DataFrame:
    """Per playoff game: each side's shrunk residual over its EARLIER games this postseason."""
    g = pgames.sort_values(["tipoff_utc", "game_id"])
    hist: dict[int, list[float]] = {}
    rows = []
    for r in g.itertuples(index=False):
        h, a = int(r.home_team_id), int(r.away_team_id)
        fh = sum(hist.get(h, [])) / (len(hist.get(h, [])) + FORM_PRIOR_GAMES)
        fa = sum(hist.get(a, [])) / (len(hist.get(a, [])) + FORM_PRIOR_GAMES)
        rows.append({"game_id": r.game_id, "home_po_form": fh, "away_po_form": fa})
        exp = theta.get(h, 0.0) - theta.get(a, 0.0) + (0.0 if r.is_neutral else hca)
        res = float(r.net_rating_margin) - exp
        hist.setdefault(h, []).append(res)
        hist.setdefault(a, []).append(-res)
    out = pd.DataFrame(rows)
    out["d_po_form"] = out.home_po_form - out.away_po_form
    return out


def augment_blocks(
    blocks: dict,
    store: RawStore,
    d: dict,
    pv: pd.DataFrame,
    all_games: pd.DataFrame,
    elo_params: dict,
    prior_playoff_pg: pd.DataFrame,
) -> dict:
    """Attach the extra team columns and po_form to every season block (rows rebuilt)."""
    from . import playoff_model as PM

    ee = elo_end(all_games, elo_params["k"], elo_params["hca"], elo_params["carry"])
    rs_pg, po_pg = d["player_games"], pd.concat([prior_playoff_pg, d["playoff_player_games"]], ignore_index=True)
    seasons = sorted(blocks, key=season_start_year)
    po_seasons_all = sorted(po_pg.season.unique(), key=season_start_year)
    adv_age = {}
    for s in seasons:
        b = blocks[s]
        earlier = [x for x in po_seasons_all if season_start_year(x) < season_start_year(s)]
        curve = playoff_share_curve(po_pg, rs_pg, [x for x in earlier if x in set(rs_pg.season)])
        mins = (
            rs_pg[rs_pg.season == s].groupby(["team_id", "player_id"]).minutes.sum().rename("rs_minutes").reset_index()
        )
        mins["team_id"] = mins.team_id.astype(int)
        v = pv[pv.season == s].set_index("player_id").lpm
        mins["value"] = mins.player_id.map(v).fillna(0.0)
        # season-end team of each player (the roster used for the playoffs)
        end_team = b["detail"][["player_id", "team_id"]].drop_duplicates("player_id").set_index("player_id").team_id
        roster = mins[mins.player_id.map(end_team) == mins.team_id]
        prev3 = [x for x in earlier[-3:]]
        prev_min = po_pg[po_pg.season.isin(prev3)].groupby("player_id").minutes.sum()
        if s not in adv_age:
            a = nba_stats.load(store, "player_advanced", s)
            adv_age[s] = a.set_index("PLAYER_ID").AGE
        extra = (
            ee[ee.season == s][["team_id", "elo_end"]]
            .merge(playoff_shift(roster, curve), on="team_id", how="right")
            .merge(experience_age(roster, prev_min, adv_age[s]), on="team_id", how="left")
            .merge(team_style(store, s)[["team_id"] + STYLE_COLS], on="team_id", how="left")
        )
        new_cols = ["elo_end", "playoff_shift", "experience", "age"] + STYLE_COLS
        teams = {tau: t.merge(extra, on="team_id", how="left") for tau, t in b["teams"].items()}
        team_cols = b["team_cols"] + [c for c in new_cols if c not in b["team_cols"]]
        theta = teams[next(iter(teams))].set_index("team_id").theta_full
        form = po_form(b["pgames"], theta, b["fit"]["coefs"][f"hca_{s}"])
        inplay = b["inplay_table"]
        rows = {
            tau: PM.playoff_rows(b["pgames"], teams[tau], inplay, team_cols).merge(
                form[["game_id", "d_po_form"]], on="game_id", how="left"
            )
            for tau in teams
        }
        b.update(
            {"teams": teams, "team_cols": team_cols, "rows": rows, "share_curve": curve, "exp_history_seasons": prev3}
        )
    return blocks
