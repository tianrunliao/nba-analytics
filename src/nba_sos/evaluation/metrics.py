"""Out-of-sample metrics and time-ordered splits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import season_start_year

EPS = 1e-15


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    df = pd.DataFrame({"bin": b, "y": y, "p": p})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
    out["bin_lo"], out["bin_hi"] = edges[out.bin], edges[out.bin + 1]
    return out[["bin_lo", "bin_hi", "n", "mean_pred", "observed"]]


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    c = calibration_table(np.asarray(y, float), np.asarray(p, float), n_bins)
    return float((c.n * (c.mean_pred - c.observed).abs()).sum() / c.n.sum())


def rolling_splits(all_seasons: list[str], first_train: str, test_seasons: list[str]) -> list[tuple[list[str], str]]:
    """Expanding window: train = seasons in [first_train, test) ; raises on any overlap."""
    out = []
    for test in test_seasons:
        train = [s for s in all_seasons
                 if season_start_year(first_train) <= season_start_year(s) < season_start_year(test)]
        if not train:
            raise ValueError(f"no training seasons before {test}")
        assert all(season_start_year(s) < season_start_year(test) for s in train)
        out.append((train, test))
    return out


def season_folds(seasons: list[str], k: int = 5) -> list[tuple[list[str], list[str]]]:
    """Grouped k-fold over seasons (train, validation), for model selection INSIDE the
    training seasons of a rolling split. Whole seasons are held out together, because games
    of one season share team strengths; folds are interleaved (season i -> fold i mod k) so
    every fold spans the eras. The final test season is never among ``seasons``."""
    s = sorted(seasons, key=season_start_year)
    k = min(k, len(s))
    if k < 2:
        raise ValueError("need at least two seasons for k-fold")
    folds = [[x for i, x in enumerate(s) if i % k == f] for f in range(k)]
    return [([x for x in s if x not in val], val) for val in folds]
