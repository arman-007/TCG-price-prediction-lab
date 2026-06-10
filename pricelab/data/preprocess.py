"""Snapshot -> modeling panel.

A *panel* is a long DataFrame on a regular period grid (monthly by default)
with one row per (card_id, period):

    period       pd.Timestamp, start of the grid period
    y            price (last observation inside the period, forward-filled
                 up to ``ffill_limit`` periods)
    log_y        np.log(y)
    is_observed  True where a real observation landed in that period
    staleness    periods since the last real observation (0 = fresh)

Semantics of forward-fill: PriceCharting prices are *standing estimates* —
the last known price IS the platform's current price until it changes, so
carrying it forward a bounded number of periods is faithful to the source.
``is_observed`` is kept so that evaluation can score forecasts only against
real observations, never against our own forward-fills.

Two data regimes exist in this DB (June 2026):
  - "deep" cohort: ~231 cards with monthly Apify backfill from 2020-2022,
    plus the recent daily sync. This is the only cohort with enough history
    for per-series models today.
  - everything else: daily sync only since 2026-05-24 (median 2 obs/card).
    Modeling those is Track 2 (global weekly model) once history accumulates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def deep_cohort_ids(prices: pd.DataFrame) -> np.ndarray:
    """Cards with Apify monthly backfill — the only deep-history cohort."""
    return prices.loc[prices["source"] == "apify", "card_id"].unique()


def build_panel(prices: pd.DataFrame, cfg) -> pd.DataFrame:
    """Build the modeling panel for the configured cohort/target/grid."""
    target = cfg.data.target_field
    freq = cfg.data.freq  # pandas *period* alias: 'M', 'W-SUN', ...

    df = prices.loc[
        prices[target].notna() & (prices[target] > 0), ["card_id", "date", target]
    ].rename(columns={target: "price"})

    if cfg.data.cohort == "deep":
        df = df[df["card_id"].isin(deep_cohort_ids(prices))]
    elif cfg.data.cohort != "all":
        raise ValueError(f"unknown cohort: {cfg.data.cohort!r}")

    if df.empty:
        raise ValueError("no rows left after target/cohort filtering")

    # Last observation per (card, period).
    df["period"] = df["date"].dt.to_period(freq)
    df = df.sort_values(["card_id", "date"])
    obs = df.groupby(["card_id", "period"], sort=False).tail(1)[
        ["card_id", "period", "price"]
    ]

    # Regular grid per card: first observed period -> global last period, so
    # every card has a row at any backtest cutoff after its first observation.
    last_period = obs["period"].max()
    wide = obs.pivot(index="period", columns="card_id", values="price")
    full_index = pd.period_range(wide.index.min(), last_period, freq=freq)
    wide = wide.reindex(full_index)

    observed = wide.notna()
    ffill_limit = int(cfg.data.ffill_limit)
    filled = wide.ffill(limit=ffill_limit) if ffill_limit > 0 else wide

    # Staleness in grid periods since the last real observation (vectorized
    # on period ordinals; NaN before a card's first observation).
    ordinals = full_index.asi8.astype("float64")
    ord_col = np.broadcast_to(ordinals[:, None], wide.shape)
    last_obs_ord = pd.DataFrame(
        np.where(observed.to_numpy(), ord_col, np.nan),
        index=full_index,
        columns=wide.columns,
    ).ffill()
    staleness = pd.DataFrame(ord_col, index=full_index, columns=wide.columns) - last_obs_ord

    panel = pd.DataFrame(
        {
            "y": filled.stack(),
            "is_observed": observed.stack(),
            "staleness": staleness.stack(),
        }
    ).reset_index()
    panel.columns = ["period", "card_id", "y", "is_observed", "staleness"]
    # Rows before a card's first observation (y NaN, staleness NaN) are noise.
    panel = panel[panel["staleness"].notna()]
    # Beyond the ffill limit y is NaN but the row must remain on the grid so
    # gaps stay visible to models that can handle missing values (SARIMAX).
    panel["log_y"] = np.log(panel["y"])

    panel = _filter_cards(panel, cfg)

    panel["period"] = panel["period"].dt.to_timestamp()
    panel = panel.sort_values(["card_id", "period"]).reset_index(drop=True)
    return panel[["card_id", "period", "y", "log_y", "is_observed", "staleness"]]


def _filter_cards(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Drop cards that can't support modeling: too few real observations, or
    prices so low that PriceCharting's $0.25-style quantization dominates.
    Optional max_cards cap keeps the most-observed cards (deterministic) for
    fast experiment iterations."""
    g = panel[panel["is_observed"]].groupby("card_id")["y"]
    n_obs = g.count()
    median_price = g.median()
    keep = n_obs[
        (n_obs >= int(cfg.data.min_obs)) & (median_price >= float(cfg.data.min_price))
    ]
    max_cards = cfg.data.get("max_cards")
    if max_cards:
        keep = keep.sort_values(ascending=False, kind="stable").head(int(max_cards))
    return panel[panel["card_id"].isin(keep.index)]


def panel_summary(panel: pd.DataFrame) -> dict:
    obs = panel[panel["is_observed"]]
    per_card = obs.groupby("card_id").size()
    return {
        "cards": int(panel["card_id"].nunique()),
        "grid_rows": int(len(panel)),
        "observed_rows": int(len(obs)),
        "first_period": str(panel["period"].min().date()),
        "last_period": str(panel["period"].max().date()),
        "obs_per_card_median": float(per_card.median()),
        "obs_per_card_min": int(per_card.min()),
        "obs_per_card_max": int(per_card.max()),
    }
