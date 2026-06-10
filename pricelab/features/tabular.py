"""Leakage-safe tabular features for the global GBM.

Contract: every feature value at row (card, t) is a function of data with
period <= t only. This is mutation-tested in tests/test_no_leakage.py —
if you add a feature, the test will catch you if it peeks forward.

Targets are h-step-ahead log returns (log_y[t+h] - log_y[t]): stationary
across cards whose price levels span 3 orders of magnitude, and trivially
mapped back to price space at prediction time.

Note on scale: groupby->transform(rolling) lambdas are fine for the deep
cohort (~200 cards). For the Track-2 full catalog (~600k cards) switch to
groupby.rolling / pivot-based variants before reusing this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def build_features(
    panel: pd.DataFrame,
    cards_meta: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    windows: tuple[int, ...] = (3, 6, 12),
) -> tuple[pd.DataFrame, list[str]]:
    """Return (feature frame, feature column names).

    The frame keeps card_id / period / log_y / is_observed so the backtest
    can mask by cutoff and join forecasts back.
    """
    df = panel.sort_values(["card_id", "period"]).reset_index(drop=True).copy()
    g = df.groupby("card_id", sort=False)
    feature_cols: list[str] = []

    # One-period log return and its lags / aggregates.
    df["r1"] = g["log_y"].diff(1)
    for k in lags:
        col = f"ret_{k}"
        df[col] = g["log_y"].diff(k)
        feature_cols.append(col)

    grp_r1 = df.groupby("card_id", sort=False)["r1"]
    grp_level = df.groupby("card_id", sort=False)["log_y"]
    for w in windows:
        mp = max(2, w // 2)
        df[f"r1_mean_{w}"] = grp_r1.transform(lambda s: s.rolling(w, min_periods=mp).mean())
        df[f"r1_std_{w}"] = grp_r1.transform(lambda s: s.rolling(w, min_periods=mp).std())
        df[f"mom_{w}"] = df["log_y"] - grp_level.transform(
            lambda s: s.rolling(w, min_periods=mp).mean()
        )
        feature_cols += [f"r1_mean_{w}", f"r1_std_{w}", f"mom_{w}"]

    # Position inside the trailing 12-period price range (0 = at low, 1 = at high).
    roll_max = grp_level.transform(lambda s: s.rolling(12, min_periods=6).max())
    roll_min = grp_level.transform(lambda s: s.rolling(12, min_periods=6).min())
    span = (roll_max - roll_min).where(lambda s: s > EPS)
    df["range_pos_12"] = ((df["log_y"] - roll_min) / span).fillna(0.5)
    feature_cols.append("range_pos_12")

    # Level, freshness, calendar.
    df["log_level"] = df["log_y"]
    df["staleness_f"] = df["staleness"].astype("float64")
    month = df["period"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    feature_cols += ["log_level", "staleness_f", "month_sin", "month_cos"]

    # Card metadata + same-period cross-sectional market move. The genre
    # median of r1 at period t only uses information available at t.
    genre = cards_meta.set_index("card_id")["genre"].fillna("unknown")
    df["genre"] = df["card_id"].map(genre).fillna("unknown").astype("category")
    df["xs_ret_genre"] = df.groupby(["genre", "period"], observed=True)["r1"].transform(
        "median"
    )
    feature_cols += ["genre", "xs_ret_genre"]

    return df, feature_cols


def add_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """h-step-ahead log-return targets + whether the target period was a real
    observation (training on forward-filled targets teaches a no-change bias,
    so the GBM trains only where target_observed_h is True)."""
    g = df.groupby("card_id", sort=False)
    for h in horizons:
        df[f"target_{h}"] = g["log_y"].shift(-h) - df["log_y"]
        df[f"target_observed_{h}"] = (
            g["is_observed"].shift(-h).astype("boolean").fillna(False).astype(bool)
        )
    return df
