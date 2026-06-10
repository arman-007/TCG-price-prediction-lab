"""Prophet wrapper.

Prophet fits a trend + seasonality curve to timestamps, so it tolerates the
irregular observation pattern better than ARIMA-family models. On this data:
  - fit on observed (non-NaN) grid points only, log scale;
  - weekly/daily seasonality off (monthly grid); yearly seasonality 'auto'
    enables itself only with >= 2 years of history;
  - quantiles come from predictive_samples (includes observation noise),
    which gives arbitrary quantiles instead of just the interval_width pair.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import ForecastResult, LocalForecaster

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


class ProphetForecaster(LocalForecaster):
    name = "prophet"

    def __init__(self, params=None, quantiles=(0.1, 0.5, 0.9), seed: int = 0, freq: str = "MS"):
        super().__init__(params, quantiles, seed)
        self.freq = freq  # pandas *offset* alias for future timestamps

    def fit(self, y: pd.Series) -> "ProphetForecaster":
        from prophet import Prophet

        # cmdstanpy (re)configures its logger at import time, which happens
        # lazily inside worker processes — quiet it here, not just at module
        # import, or every fit logs two INFO lines.
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        obs = y.dropna()
        if len(obs) < 10:
            raise ValueError(f"too short for Prophet: {len(obs)}")
        df = pd.DataFrame({"ds": obs.index, "y": obs.to_numpy()})

        n_changepoints = min(int(self.params.get("n_changepoints", 15)), max(1, len(df) // 3))
        self.model = Prophet(
            growth=self.params.get("growth", "linear"),
            n_changepoints=n_changepoints,
            changepoint_prior_scale=float(self.params.get("changepoint_prior_scale", 0.05)),
            changepoint_range=float(self.params.get("changepoint_range", 0.9)),
            yearly_seasonality=self.params.get("yearly_seasonality", "auto"),
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_prior_scale=float(self.params.get("seasonality_prior_scale", 1.0)),
            seasonality_mode="additive",  # additive on log == multiplicative on price
            interval_width=0.8,
            uncertainty_samples=int(self.params.get("uncertainty_samples", 300)),
        )
        self.model.fit(df)
        self.last_ds = obs.index[-1]
        return self

    def predict(self, horizon: int) -> ForecastResult:
        future = pd.DataFrame(
            {"ds": pd.date_range(self.last_ds, periods=horizon + 1, freq=self.freq)[1:]}
        )
        point = self.model.predict(future)["yhat"].to_numpy()
        samples = self.model.predictive_samples(future)["yhat"]  # (horizon, n_samples)
        qmap = {q: np.quantile(samples, q, axis=1) for q in self.quantiles}
        return ForecastResult(point, qmap, self.name).sorted_quantiles()
