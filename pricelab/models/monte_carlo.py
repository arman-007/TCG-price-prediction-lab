"""Monte Carlo path simulation via moving-block bootstrap of log returns.

Why block bootstrap instead of Gaussian GBM: collectible returns are fat-
tailed and weakly autocorrelated (hype runs / corrections). Resampling
*blocks* of real historical returns preserves both, where a normal
distribution would understate tail risk.

Roles in this lab:
  1. standalone probabilistic forecaster (a "stochastic naive-with-history");
  2. the uncertainty engine the ensemble can lean on when member intervals
     disagree (quantiles from simulated paths are always available).

The drift knob matters: raw resampling inherits the historical mean return,
which over-extrapolates after a pump. drift="half" shrinks it 50% toward
zero, drift="none" removes it (pure uncertainty around today's price).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastResult, LocalForecaster, residual_sigma


class MonteCarloForecaster(LocalForecaster):
    name = "monte_carlo"

    # Per-step |log return| cap: collectible data glitches (e.g. a $0.99
    # mislisting) otherwise enter the resampling pool and blow up paths.
    MAX_ABS_RETURN = 1.1

    def fit(self, y: pd.Series) -> "MonteCarloForecaster":
        obs = y.dropna()
        if obs.empty:
            raise ValueError("empty series")
        self.last = float(obs.iloc[-1])
        r = np.diff(obs.to_numpy())
        r = np.clip(r, -self.MAX_ABS_RETURN, self.MAX_ABS_RETURN)

        drift_mode = self.params.get("drift", "half")
        if len(r) >= 4 and drift_mode != "full":
            shrink = 0.0 if drift_mode == "none" else 0.5
            r = r - shrink * r.mean()
        self.returns = r if len(r) >= 4 else None
        self.fallback_sigma = residual_sigma(y)
        # self.seed is already card-specific (harness derives it via
        # utils.stable_card_seed), so runs are reproducible across workers.
        self.rng = np.random.default_rng(self.seed)
        return self

    def predict(self, horizon: int) -> ForecastResult:
        n_paths = int(self.params.get("n_paths", 2000))
        block = max(1, int(self.params.get("block_size", 3)))

        if self.returns is None:
            # Too little history to resample: Gaussian zero-drift fallback.
            steps = self.rng.normal(0.0, self.fallback_sigma, size=(n_paths, horizon))
        else:
            r = self.returns
            block = min(block, len(r))
            n_blocks = int(np.ceil(horizon / block))
            starts = self.rng.integers(0, len(r) - block + 1, size=(n_paths, n_blocks))
            idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)
            steps = r[idx[:, :horizon]]

        paths = self.last + np.cumsum(steps, axis=1)
        mean = np.median(paths, axis=0)
        qmap = {q: np.quantile(paths, q, axis=0) for q in self.quantiles}
        return ForecastResult(mean, qmap, self.name).sorted_quantiles()
