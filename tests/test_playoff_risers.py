"""Playoff risers: same-scale RS/PO box value, playoff shift, opponent adjustment, shrinkage."""

import numpy as np
import pandas as pd
import pytest

from nba_sos.ratings import player_value as V
from nba_sos.ratings import playoff_risers as PR

INPUTS = [f for f in PR.BOX if f != "ts_x_usg"] + ["ts"]


def _rs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: rng.normal(10, 3, n) for c in INPUTS})
    d["ts"] = rng.normal(0.56, 0.04, n)
    d["usg"] = rng.normal(0.2, 0.04, n)
    d["player_id"] = range(n)
    d["player_name"] = [f"p{i}" for i in range(n)]
    d["min_total"] = rng.uniform(300, 3000, n)
    return d


def _beta(seed=1):
    rng = np.random.default_rng(seed)
    return dict(zip(PR.BOX, rng.normal(0, 1, len(PR.BOX))))


def test_rs_value_matches_player_value_standardisation():
    rs = _rs()
    beta = _beta()
    po = rs.copy()
    out = PR.season_deltas(rs, po, beta)
    # same z-scores as player_value.standardise (no on/off inputs needed for box features)
    z = V.standardise(
        rs.assign(season="s", on_court_net=0.0),
        pd.DataFrame({"season": ["s"], "player_id": [0], "on_off48": [0.0]}),
        pd.DataFrame({"season": ["s"], "player_id": [0], "team_net": [0.0]}),
    )
    expect = z[[f"z_{f}" for f in PR.BOX]].to_numpy() @ np.array([beta[f] for f in PR.BOX])
    np.testing.assert_allclose(out.rs_box, expect, atol=1e-9)
    np.testing.assert_allclose(out.delta_raw, 0.0, atol=1e-9)  # no change -> no delta


def test_common_playoff_change_is_removed():
    rs = _rs()
    po = rs.copy()
    po["pts_100"] -= 3.0  # everyone scores less vs playoff defences
    po["min_total"] = 400.0
    po.loc[0, "pts_100"] += 6.0  # player 0: +3 vs RS, i.e. +6 vs the typical change
    out = PR.season_deltas(rs, po, _beta()).set_index("player_id")
    norms = PR.season_norms(rs)
    shift = (-3.0 * 39 + 3.0) / 40 / norms["sd"]["pts_100"]
    b = _beta()["pts_100"]
    assert out.delta_raw[0] == pytest.approx((3.0 / norms["sd"]["pts_100"] - shift) * b)
    assert out.delta_raw[1] == pytest.approx((-3.0 / norms["sd"]["pts_100"] - shift) * b)
    assert np.average(out.delta_raw, weights=out.po_min) == pytest.approx(0.0, abs=1e-9)


def test_opponent_adjustment_removes_linear_opponent_effect():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame(
        {"season": np.repeat(["a", "b"], n // 2), "po_min": rng.uniform(150, 900, n), "opp_theta": rng.normal(3, 2, n)}
    )
    d["delta_raw"] = -0.5 * d.opp_theta + rng.normal(0, 0.1, n)
    out, slope = PR.opponent_adjust(d)
    assert slope == pytest.approx(-0.5, abs=0.02)
    xc = out.opp_theta - out.groupby("season").opp_theta.transform("mean")
    assert abs(np.corrcoef(xc, out.delta)[0, 1]) < 0.1


def test_noise_model_recovers_persistent_variance_and_shrinks_by_minutes():
    rng = np.random.default_rng(3)
    tau, sigma = 1.0, 30.0
    rows = []
    for p in range(600):
        mu = rng.normal(0, tau)
        for s in range(4):
            m = rng.uniform(200, 900)
            rows.append(
                {
                    "player_id": p,
                    "player_name": f"p{p}",
                    "season": f"{2000 + s}",
                    "po_min": m,
                    "delta": mu + rng.normal(0, sigma / np.sqrt(m)),
                    "rs_box": 0.0,
                    "d_pts_100": 0.0,
                    "d_ts": 0.0,
                    "d_usg": 0.0,
                }
            )
    d = pd.DataFrame(rows)
    nm = PR.noise_model(d)
    assert nm["tau2"] == pytest.approx(tau**2, rel=0.25)
    assert nm["sigma2"] == pytest.approx(sigma**2, rel=0.2)
    c = PR.career(d, nm, min_minutes=0).set_index("player_id")
    g = d[d.player_id == 0]
    raw = np.average(g.delta, weights=g.po_min)
    assert c.loc[0, "est"] == pytest.approx(raw * g.po_min.sum() / (nm["k_minutes"] + g.po_min.sum()))
    assert (c.est.abs() <= c.raw_delta.abs() + 1e-12).all()
