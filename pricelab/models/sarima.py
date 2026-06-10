"""SARIMA via statsmodels SARIMAX with a small AIC-selected order grid.

Design notes:
  - Runs on the log-price grid series; SARIMAX's Kalman filter handles the
    NaN gaps natively, so we never have to fake values for it.
  - pmdarima (auto_arima) is deliberately avoided: it lags numpy releases
    and its search is overkill for <=80-point monthly series. A small
    explicit grid is transparent and fast.
  - Order selection is O(grid) per call; the backtest reuses the order
    chosen at a card's first cutoff for later cutoffs (leakage-free, since
    earlier training windows are subsets of later ones) via `fixed_order`.
  - Seasonality off by default: monthly collectible series here are short
    (40-70 points) and trend/momentum dominated; a (P,D,Q,12) term is a
    tuning option, not a default.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastResult, LocalForecaster, gaussian_quantiles


def _candidate_orders(params: dict) -> list[tuple[tuple[int, int, int], str]]:
    max_p = int(params.get("max_p", 2))
    max_q = int(params.get("max_q", 2))
    d_options = params.get("d_options", [0, 1])
    candidates = []
    for d in d_options:
        # d=0: constant mean level; d=1: 'c' becomes drift. Both trends are
        # worth trying for trending collectibles.
        trends = ["c"] if d == 0 else ["n", "c"]
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0 and d == 0:
                    continue  # white noise around a constant — naive covers it
                for trend in trends:
                    candidates.append(((p, d, q), trend))
    return candidates


def _fit_one(values: np.ndarray, order, trend, seasonal_order=(0, 0, 0, 0)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            values,
            order=order,
            trend=trend,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        return model.fit(disp=False, maxiter=100)


class SarimaForecaster(LocalForecaster):
    name = "sarima"

    def fit(self, y: pd.Series) -> "SarimaForecaster":
        # Keep internal NaNs for the Kalman filter, but trim the unobserved
        # tail: SARIMAX would otherwise condition the forecast start on pure
        # missingness beyond the last real price.
        y = y.loc[: y.last_valid_index()]
        values = y.to_numpy(dtype="float64")
        if np.isfinite(values).sum() < 10:
            raise ValueError("too short for SARIMA")

        seasonal = bool(self.params.get("seasonal", False))
        m = int(self.params.get("m", 12))
        seasonal_order = tuple(self.params.get("seasonal_order", (1, 0, 1, m))) if seasonal else (0, 0, 0, 0)

        fixed = self.params.get("fixed_order")  # ((p,d,q), trend) from a prior cutoff
        if fixed is not None:
            order, trend = (tuple(fixed[0]), fixed[1])
            self.result = _fit_one(values, order, trend, seasonal_order)
            self.selected = (order, trend)
            return self

        best, best_aic = None, np.inf
        for order, trend in _candidate_orders(self.params):
            try:
                res = _fit_one(values, order, trend, seasonal_order)
            except Exception:
                continue
            if np.isfinite(res.aic) and res.aic < best_aic:
                best, best_aic = (res, (order, trend)), res.aic
        if best is None:
            raise RuntimeError("no SARIMA candidate converged")
        self.result, self.selected = best
        return self

    def predict(self, horizon: int) -> ForecastResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.result.get_forecast(steps=horizon)
            mean = np.asarray(pred.predicted_mean, dtype="float64")
            se = np.asarray(pred.se_mean, dtype="float64")
        se = np.where(np.isfinite(se), se, np.nanmax(se[np.isfinite(se)]) if np.isfinite(se).any() else 0.1)
        return gaussian_quantiles(mean, se, self.quantiles, self.name)
