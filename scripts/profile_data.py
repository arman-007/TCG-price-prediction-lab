#!/usr/bin/env python
"""Profile the snapshot: data regimes, cohort depth, cadence, price scales.

Writes reports/data_profile.md (+ figures) — rerun after each snapshot
refresh; modeling decisions in configs/ should trace back to this report.
"""

from pricelab.threadlimits import limit_blas_threads

limit_blas_threads(1)

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricelab.config import load_config
from pricelab.data.preprocess import build_panel, deep_cohort_ids, panel_summary
from pricelab.data.snapshot import load_snapshot
from pricelab.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    prices, cards = load_snapshot(cfg.data.snapshot_dir)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    target = cfg.data.target_field
    deep_ids = deep_cohort_ids(prices)
    obs_per_card = prices.groupby("card_id").size()

    lines = [
        "# Data profile",
        "",
        f"snapshot: `{cfg.data.snapshot_dir}` | target field: `{target}`",
        "",
        "## Regimes",
        "",
        f"- price rows: {len(prices):,} across {prices['card_id'].nunique():,} cards",
        f"- date range: {prices['date'].min().date()} .. {prices['date'].max().date()}"
        f" ({prices['date'].nunique()} distinct dates)",
        f"- sources: {prices['source'].value_counts().to_dict()}",
        f"- deep cohort (apify monthly backfill): {len(deep_ids)} cards",
        f"- obs/card overall: median {obs_per_card.median():.0f}, p90 "
        f"{obs_per_card.quantile(0.9):.0f}, max {obs_per_card.max()}",
        "",
    ]

    # Observation cadence: rows per date reveals the monthly-then-daily regimes.
    per_date = prices.groupby("date").size()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.semilogy(per_date.index, per_date.values, marker=".", lw=0.5)
    ax.set_title("price rows per date (log scale) — monthly backfill era vs daily sync era")
    fig.tight_layout()
    fig.savefig(out / "figures" / "rows_per_date.png", dpi=120)
    plt.close(fig)

    # Deep-cohort modeling panel as configured.
    panel = build_panel(prices, cfg)
    summary = panel_summary(panel)
    lines += ["## Modeling panel (as configured)", ""]
    lines += [f"- {k}: {v}" for k, v in summary.items()]
    lines.append("")

    obs = panel[panel["is_observed"]]
    med_price = obs.groupby("card_id")["y"].median()
    lines += [
        "## Price scale (panel cards, median observed price)",
        "",
        f"- p10 ${med_price.quantile(0.1):.2f} | p50 ${med_price.median():.2f} | "
        f"p90 ${med_price.quantile(0.9):.2f} | max ${med_price.max():.2f}",
        "",
        "![rows per date](figures/rows_per_date.png)",
        "",
        "![sample series](figures/sample_series.png)",
        "",
    ]

    # Six sample series across the liquidity range.
    sample_ids = (
        obs.groupby("card_id").size().sort_values(ascending=False).index[::40][:6]
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    names = cards.set_index("card_id")["product_name"]
    for ax, cid in zip(axes.ravel(), sample_ids):
        g = panel[panel["card_id"] == cid]
        ax.plot(g["period"], g["y"], lw=1)
        ax.scatter(g.loc[g["is_observed"], "period"], g.loc[g["is_observed"], "y"], s=8)
        ax.set_title(str(names.get(cid, cid))[:38], fontsize=8)
        ax.set_yscale("log")
    fig.suptitle("sample deep-cohort series (dots = real observations, line = grid w/ ffill)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "sample_series.png", dpi=120)
    plt.close(fig)

    (out / "data_profile.md").write_text("\n".join(lines))
    logger.info("wrote %s", out / "data_profile.md")


if __name__ == "__main__":
    main()
