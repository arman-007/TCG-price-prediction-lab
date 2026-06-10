#!/usr/bin/env python
"""Produce CURRENT forecasts (P10/P50/P90 prices) from the latest grid period.

Reuses the exact backtest machinery with a single cutoff at the last grid
period and an artificially extended period axis, so production forecasts go
through the same code path that was validated by the backtest.

Ensemble weights come from a prior backtest run (--weights <run>/weights.json);
without them, members blend with equal weights (and the report says so).

Examples:
    python scripts/forecast.py --weights experiments/<run>/weights.json
    python scripts/forecast.py --champion prophet
"""

from pricelab.threadlimits import limit_blas_threads

limit_blas_threads(1)

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from pricelab.config import load_config, merge_into
from pricelab.data.preprocess import build_panel
from pricelab.data.snapshot import load_snapshot
from pricelab.evaluation.backtest import (
    _compute_card_stats,
    _run_card,
    _run_global_xgb,
    blend_forecasts,
    period_freq_to_offset,
)
from pricelab.evaluation.metrics import qcol
from pricelab.models.registry import split_model_names
from pricelab.utils import set_seeds, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default=None, help="weights.json from a backtest run")
    parser.add_argument("--champion", default="ensemble",
                        help="model whose forecasts go into the CSV (default: ensemble)")
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--out", default="experiments/forecasts_latest")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config, args.overrides)
    if args.tuned:
        for path in sorted(Path("configs/tuned").glob("*.yaml")):
            merge_into(cfg, yaml.safe_load(path.read_text()))
    set_seeds(int(cfg.seed))

    prices, cards = load_snapshot(cfg.data.snapshot_dir)
    panel = build_panel(prices, cfg)

    quantiles = tuple(float(q) for q in cfg.quantiles)
    horizons = [int(h) for h in cfg.horizons]
    local_names, global_names, _ = split_model_names(list(cfg.models))
    freq_offset = period_freq_to_offset(str(cfg.data.freq))

    periods = sorted(pd.Timestamp(p) for p in panel["period"].unique())
    cutoff = periods[-1]
    future = list(pd.date_range(cutoff, periods=max(horizons) + 1, freq=freq_offset)[1:])
    periods_ext = periods + future
    logger.info("forecasting from %s for %d cards", cutoff.date(), panel["card_id"].nunique())

    model_cfgs = {n: dict(cfg.get(n, {}) or {}) for n in local_names}
    rows: list[dict] = []
    if local_names:
        tasks = [
            (cid, g.set_index("period")["log_y"], g.set_index("period")["is_observed"])
            for cid, g in panel.groupby("card_id")
        ]
        results = Parallel(n_jobs=int(cfg.runtime.n_jobs), prefer="processes")(
            delayed(_run_card)(
                cid, y, obs, [cutoff], periods_ext, local_names, model_cfgs,
                quantiles, int(cfg.seed), int(cfg.backtest.min_train_obs),
                horizons, freq_offset,
            )
            for cid, y, obs in tasks
        )
        for r in results:
            rows.extend(r)
    if "xgboost" in global_names:
        card_stats = _compute_card_stats(panel, [cutoff])
        rows.extend(
            _run_global_xgb(panel, cards, cfg, [cutoff], periods_ext, quantiles,
                            horizons, card_stats)
        )
    forecasts = pd.DataFrame(rows)
    if forecasts.empty:
        raise SystemExit("no forecasts produced — check filters")

    # Ensemble from saved (or equal) weights.
    members = [m for m in cfg.ensemble.members if m in set(forecasts["model"])]
    if args.weights:
        raw = json.loads(Path(args.weights).read_text())
        weights = {int(h): w for h, w in raw.items()}
        logger.info("using backtest weights from %s", args.weights)
    else:
        weights = {h: {m: 1.0 / len(members) for m in members} for h in horizons}
        logger.warning("no --weights given: blending members with EQUAL weights")
    ens = blend_forecasts(forecasts[forecasts["model"].isin(members)], weights, quantiles)
    if not ens.empty:
        forecasts = pd.concat([forecasts, ens], ignore_index=True)

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_base.with_suffix(".parquet"), index=False)

    champ = forecasts[forecasts["model"] == args.champion].copy()
    if champ.empty:
        raise SystemExit(f"champion {args.champion!r} produced no rows")
    qmin, qmax = min(quantiles), max(quantiles)
    names = cards.set_index("card_id")[["product_name", "console_name"]]
    champ = champ.join(names, on="card_id")
    csv = pd.DataFrame(
        {
            "card_id": champ["card_id"],
            "product_name": champ["product_name"],
            "console_name": champ["console_name"],
            "as_of": champ["cutoff"].dt.date,
            "horizon_steps": champ["h"],
            "target_period": champ["target_period"].dt.date,
            "last_price": np.exp(champ["last_log"]).round(2),
            "pred_price": np.exp(champ["pred_log"]).round(2),
            f"p{int(qmin * 100)}_price": np.exp(champ[qcol(qmin)]).round(2),
            f"p{int(qmax * 100)}_price": np.exp(champ[qcol(qmax)]).round(2),
            "model": champ["model"],
        }
    ).sort_values(["card_id", "horizon_steps"])
    csv.to_csv(out_base.with_suffix(".csv"), index=False)
    logger.info("wrote %s (%d rows) and %s",
                out_base.with_suffix(".csv"), len(csv), out_base.with_suffix(".parquet"))


if __name__ == "__main__":
    main()
