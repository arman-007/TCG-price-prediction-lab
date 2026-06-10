"""Forecaster interfaces.

All models operate in LOG-price space end to end:
  - inputs: pd.Series of log prices indexed by grid-period timestamps,
    possibly containing NaN gaps (beyond the forward-fill limit);
  - outputs: ForecastResult with log-space mean path and quantile paths.

exp(point forecast) is therefore the *median* price under a lognormal view,
which is the robust point estimate we want for skewed collectible prices.

Two families:
  - LocalForecaster: fit per card series (naive, ETS, SARIMA, Prophet, MC).
  - global models (XGBoost) live in xgboost_model.py and fit once per
    backtest cutoff across the whole panel; the harness routes them apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
# Floor on the 1-step residual sigma (log space) so degenerate flat series
# don't produce zero-width intervals.
MIN_SIGMA = 1e-3


@dataclass
class ForecastResult:
    """Log-space forecast for grid steps 1..H ahead of the fit cutoff."""

    mean: np.ndarray
    quantiles: dict[float, np.ndarray]
    model: str
    fallback: bool = False

    def sorted_quantiles(self) -> "ForecastResult":
        """Enforce monotone quantiles (fixes quantile crossing per step)."""
        qs = sorted(self.quantiles)
        stacked = np.sort(np.vstack([self.quantiles[q] for q in qs]), axis=0)
        return ForecastResult(
            self.mean, {q: stacked[i] for i, q in enumerate(qs)}, self.model, self.fallback
        )


class LocalForecaster(ABC):
    """Per-series model. fit() receives log prices on the grid (may contain
    NaNs); predict() returns 1..horizon steps ahead."""

    name = "base"

    def __init__(self, params: dict | None = None, quantiles=DEFAULT_QUANTILES, seed: int = 0):
        self.params = dict(params or {})
        self.quantiles = tuple(quantiles)
        self.seed = seed

    @abstractmethod
    def fit(self, y: pd.Series) -> "LocalForecaster": ...

    @abstractmethod
    def predict(self, horizon: int) -> ForecastResult: ...


def gaussian_quantiles(
    mean: np.ndarray, se: np.ndarray, quantiles, name: str, fallback: bool = False
) -> ForecastResult:
    """Build a ForecastResult from a Gaussian mean/se path (log space)."""
    se = np.maximum(np.asarray(se, dtype="float64"), MIN_SIGMA)
    qmap = {q: mean + stats.norm.ppf(q) * se for q in quantiles}
    return ForecastResult(np.asarray(mean, dtype="float64"), qmap, name, fallback)


def residual_sigma(y: pd.Series) -> float:
    """Robust 1-step log-return sigma used by fallback interval logic."""
    r = y.dropna().diff().dropna()
    if len(r) < 3:
        return 0.08  # weakly-informative prior for collectible monthlies
    return float(max(r.std(ddof=1), MIN_SIGMA))
