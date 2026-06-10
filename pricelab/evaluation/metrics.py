"""Forecast accuracy metrics.

Primary headline metric: MASE in log space — |log error| scaled by the
card's in-sample mean |1-step log return| at the cutoff. Scale-free across
cards whose prices span $1 to $2,500+, and directly interpretable:
MASE < 1.0 means "beats the naive random walk".

sMAPE/MAE in price space are reported for product-facing intuition;
pinball loss and interval coverage grade the probabilistic forecasts,
which for a portfolio app matter as much as the point forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Floor for the MASE denominator (mean |1-step log return| in-sample).
# PriceCharting leaves many cards flat for months, so the raw scale can hit
# ~0 and a single later move produces ASE in the thousands — measured on
# this data, rows with scale<=2e-3 were 1.7% of forecasts but 96% of the
# total ASE sum. Treat <1%/period as "flat" and scale by 1% instead; the
# median MASE is reported alongside the mean as the robust view.
MIN_MASE_SCALE = 0.01


def qcol(q: float) -> str:
    return f"q{int(round(q * 100))}"


def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = actual - pred
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))


def summarize(eval_df: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Aggregate per (model, split, horizon). eval_df rows are individual
    forecasts already joined to *really observed* actuals."""
    qmin, qmax = min(quantiles), max(quantiles)
    cov_label = f"coverage_{int(round((qmax - qmin) * 100))}"
    records = []
    for (model, split, h), grp in eval_df.groupby(["model", "split", "h"]):
        err = grp["actual_log"] - grp["pred_log"]
        ase = err.abs() / grp["naive_scale"]
        price_pred = np.exp(grp["pred_log"])
        price_act = np.exp(grp["actual_log"])
        denom = (np.abs(price_act) + np.abs(price_pred)).clip(lower=1e-9)

        moved = (grp["actual_log"] - grp["last_log"]).abs() > 1e-9
        dir_acc = float(
            (
                np.sign(grp.loc[moved, "pred_log"] - grp.loc[moved, "last_log"])
                == np.sign(grp.loc[moved, "actual_log"] - grp.loc[moved, "last_log"])
            ).mean()
        ) if moved.any() else np.nan

        rec = {
            "model": model,
            "split": split,
            "h": int(h),
            "n": int(len(grp)),
            "mase": float(ase.mean()),
            "mase_median": float(ase.median()),
            "rmse_log": float(np.sqrt((err**2).mean())),
            "mae_price": float((price_act - price_pred).abs().mean()),
            "smape_price": float((200.0 * np.abs(price_act - price_pred) / denom).mean()),
            "dir_acc": dir_acc,
            cov_label: float(
                (
                    (grp["actual_log"] >= grp[qcol(qmin)])
                    & (grp["actual_log"] <= grp[qcol(qmax)])
                ).mean()
            ),
            "fallback_rate": float(grp["fallback"].mean()),
            "fit_seconds": float(grp["fit_seconds"].sum()),
        }
        for q in quantiles:
            rec[f"pinball_{qcol(q)}"] = pinball_loss(
                grp["actual_log"].to_numpy(), grp[qcol(q)].to_numpy(), q
            )
        records.append(rec)
    return pd.DataFrame.from_records(records).sort_values(["split", "h", "mase"]).reset_index(drop=True)
