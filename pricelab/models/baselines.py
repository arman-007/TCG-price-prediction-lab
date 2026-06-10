"""Baseline forecasters.

Collectible prices are close to a random walk, so the naive forecast is THE
benchmark: any of the fancy models must beat it out of sample or it has no
business shipping. MASE is defined against naive for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    ForecastResult,
    LocalForecaster,
    gaussian_quantiles,
    residual_sigma,
)


class NaiveForecaster(LocalForecaster):
    """Random walk: last observed log price, sqrt-of-time intervals."""

    name = "naive"

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        obs = y.dropna()
        if obs.empty:
            raise ValueError("empty series")
        self.last = float(obs.iloc[-1])
        self.sigma = residual_sigma(y)
        return self

    def predict(self, horizon: int) -> ForecastResult:
        steps = np.arange(1, horizon + 1)
        mean = np.full(horizon, self.last)
        se = self.sigma * np.sqrt(steps)
        return gaussian_quantiles(mean, se, self.quantiles, self.name)


class DriftForecaster(LocalForecaster):
    """Random walk with drift, with optional damping (phi < 1 keeps the
    long-horizon extrapolation from running away on short pumped series)."""

    name = "drift"

    def fit(self, y: pd.Series) -> "DriftForecaster":
        obs = y.dropna()
        if obs.empty:
            raise ValueError("empty series")
        self.last = float(obs.iloc[-1])
        n = len(obs)
        self.slope = (float(obs.iloc[-1]) - float(obs.iloc[0])) / (n - 1) if n > 1 else 0.0
        self.phi = float(self.params.get("damping", 1.0))
        self.sigma = residual_sigma(y)
        return self

    def predict(self, horizon: int) -> ForecastResult:
        steps = np.arange(1, horizon + 1)
        if self.phi < 1.0:
            cum = self.phi * (1 - self.phi**steps) / (1 - self.phi)
        else:
            cum = steps.astype("float64")
        mean = self.last + self.slope * cum
        se = self.sigma * np.sqrt(steps)
        return gaussian_quantiles(mean, se, self.quantiles, self.name)


class SeasonalNaiveForecaster(LocalForecaster):
    """y[T+h] = y[T+h-m]; falls back to naive when history < one season."""

    name = "seasonal_naive"

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        self.m = int(self.params.get("m", 12))
        obs = y.dropna()
        if obs.empty:
            raise ValueError("empty series")
        self.values = obs.to_numpy()
        self.sigma = residual_sigma(y)
        return self

    def predict(self, horizon: int) -> ForecastResult:
        steps = np.arange(1, horizon + 1)
        if len(self.values) <= self.m:
            mean = np.full(horizon, self.values[-1])
        else:
            mean = np.array(
                [self.values[len(self.values) - self.m + ((h - 1) % self.m)] for h in steps]
            )
        se = self.sigma * np.sqrt(steps)
        return gaussian_quantiles(mean, se, self.quantiles, self.name)


class EtsForecaster(LocalForecaster):
    """Damped-trend exponential smoothing (statsmodels ETSModel, additive on
    log prices) with model-based prediction intervals."""

    name = "ets"

    def fit(self, y: pd.Series) -> "EtsForecaster":
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        obs = y.dropna()
        if len(obs) < 8:
            raise ValueError(f"too short for ETS: {len(obs)}")
        trend = self.params.get("trend", "add")
        damped = bool(self.params.get("damped", True))
        # RangeIndex Series, not ndarray: ETSModel.get_prediction requires a
        # pandas object (it reads .index off the prediction), and a dateless
        # positional index sidesteps no-frequency DatetimeIndex warnings.
        model = ETSModel(
            pd.Series(obs.to_numpy()),
            error="add",
            trend=trend if trend != "none" else None,
            damped_trend=damped if trend != "none" else False,
            seasonal=None,
        )
        self.result = model.fit(disp=False)
        self.n = len(obs)
        return self

    def predict(self, horizon: int) -> ForecastResult:
        pred = self.result.get_prediction(start=self.n, end=self.n + horizon - 1)
        frame = pred.summary_frame(alpha=0.2)  # 80% band -> recover se
        mean = frame["mean"].to_numpy()
        z80 = 1.2815515655446004
        se = (frame["pi_upper"].to_numpy() - frame["pi_lower"].to_numpy()) / (2 * z80)
        return gaussian_quantiles(mean, se, self.quantiles, self.name)
