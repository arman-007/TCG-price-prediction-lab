"""Global gradient-boosted model, pooled across all cards.

Rationale: with ~200 deep series of 40-80 monthly points each, a per-card ML
model is hopeless, but a GLOBAL model sees thousands of (card, month) rows
and can learn cross-card regularities (mean reversion after spikes, genre
momentum, volatility clustering). This is the standard play for "many
related short series" and the main hypothesis this lab exists to test.

Strategy:
  - DIRECT multi-horizon: one regressor per horizon h, predicting the
    h-step log return. No recursive feedback, no error accumulation.
  - One model per horizon trained with multi-quantile pinball loss
    (objective='reg:quantileerror', xgboost >= 2.0): P10/P50/P90 in a
    single fit. The P50 path doubles as the point forecast, consistent
    with the log-median convention used everywhere else in the lab.
  - Leakage discipline: features at row t use data <= t (see features/
    tabular.py); training rows at cutoff C satisfy t + h <= C, so no
    target reaches past the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastResult

DEFAULT_PARAMS = {
    "max_depth": 3,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "min_child_weight": 5.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
}


class GlobalXGBoost:
    name = "xgboost"

    def __init__(self, params: dict, horizons: list[int], quantiles=(0.1, 0.5, 0.9), seed: int = 0):
        self.cfg = dict(params or {})
        self.horizons = list(horizons)
        self.quantiles = sorted(quantiles)
        self.seed = seed
        self.models: dict[int, object] = {}

    def _make_regressor(self):
        import xgboost as xgb

        params = {**DEFAULT_PARAMS, **self.cfg.get("params", {})}
        return xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.asarray(self.quantiles, dtype="float64"),
            tree_method="hist",
            enable_categorical=True,
            random_state=self.seed,
            n_jobs=int(self.cfg.get("n_jobs", 0) or 0) or None,
            **params,
        )

    def fit(self, feat: pd.DataFrame, feature_cols: list[str], cutoff_idx: int) -> "GlobalXGBoost":
        """Train one model per horizon on rows with period_idx <= cutoff - h
        and a *really observed* target (never on forward-filled echoes)."""
        self.feature_cols = list(feature_cols)
        self.models = {}
        for h in self.horizons:
            mask = (
                (feat["period_idx"] <= cutoff_idx - h)
                & feat[f"target_{h}"].notna()
                & feat[f"target_observed_{h}"]
                & feat["ret_1"].notna()  # require at least minimal history
            )
            train = feat.loc[mask]
            if len(train) < 200:
                continue  # not enough pooled data for this horizon yet
            reg = self._make_regressor()
            reg.fit(train[self.feature_cols], train[f"target_{h}"])
            self.models[h] = reg
        if not self.models:
            raise RuntimeError("no horizon had enough training rows")
        return self

    def predict_at_cutoff(self, feat: pd.DataFrame, cutoff_idx: int) -> dict[int, ForecastResult]:
        """Forecast every card with a feature row exactly at the cutoff.
        Returns {card_id: ForecastResult} with paths over max(horizons)."""
        rows = feat[(feat["period_idx"] == cutoff_idx) & feat["ret_1"].notna()]
        if rows.empty:
            return {}
        max_h = max(self.horizons)
        n = len(rows)
        # Pre-fill with NaN; only configured horizons get real values.
        mean = np.full((n, max_h), np.nan)
        qpaths = {q: np.full((n, max_h), np.nan) for q in self.quantiles}

        base = rows["log_y"].to_numpy()
        for h, reg in self.models.items():
            pred = np.asarray(reg.predict(rows[self.feature_cols]))  # (n, n_quantiles)
            if pred.ndim == 1:  # single-quantile edge case
                pred = pred[:, None]
            pred = np.sort(pred, axis=1)  # quantile crossing guard
            for qi, q in enumerate(self.quantiles):
                qpaths[q][:, h - 1] = base + pred[:, qi]
            median_col = self.quantiles.index(0.5) if 0.5 in self.quantiles else pred.shape[1] // 2
            mean[:, h - 1] = base + pred[:, median_col]

        out = {}
        for i, card_id in enumerate(rows["card_id"].to_numpy()):
            out[int(card_id)] = ForecastResult(
                mean[i], {q: qpaths[q][i] for q in self.quantiles}, self.name
            )
        return out
