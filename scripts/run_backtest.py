#!/usr/bin/env python
"""Run the rolling-origin model bake-off and write an experiment report.

Examples:
    python scripts/run_backtest.py --fast --tag smoke
    python scripts/run_backtest.py --tag full
    python scripts/run_backtest.py --tuned --tag full-tuned
    python scripts/run_backtest.py --models naive,prophet --set backtest.n_cutoffs=12
"""

from pricelab.threadlimits import limit_blas_threads

limit_blas_threads(1)

import argparse
import logging
from pathlib import Path

import yaml

from pricelab.config import load_config, merge_into, set_dotted
from pricelab.data.preprocess import build_panel
from pricelab.data.snapshot import load_snapshot
from pricelab.evaluation.backtest import run_backtest
from pricelab.utils import set_seeds, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", help="comma-separated subset, e.g. naive,prophet")
    parser.add_argument("--tuned", action="store_true", help="merge configs/tuned/*.yaml")
    parser.add_argument("--fast", action="store_true",
                        help="smoke mode: fewer cutoffs/cards for a quick end-to-end pass")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="key.path=value")
    parser.add_argument("--out", default="experiments")
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config, args.overrides)
    if args.tuned:
        for path in sorted(Path("configs/tuned").glob("*.yaml")):
            merge_into(cfg, yaml.safe_load(path.read_text()))
            logger.info("merged tuned params from %s", path)
    if args.models:
        cfg["models"] = [m.strip() for m in args.models.split(",")]
    if args.fast:
        set_dotted(cfg, "backtest.n_cutoffs", 6)
        set_dotted(cfg, "data.max_cards", 60)
        set_dotted(cfg, "monte_carlo.n_paths", 500)
        set_dotted(cfg, "prophet.uncertainty_samples", 150)

    set_seeds(int(cfg.seed))
    prices, cards = load_snapshot(cfg.data.snapshot_dir)
    panel = build_panel(prices, cfg)
    logger.info("panel: %d cards, %d rows", panel["card_id"].nunique(), len(panel))

    result = run_backtest(panel, cards, cfg, out_dir=args.out, tag=args.tag)
    metrics = result["metrics"]
    test = metrics[metrics["split"] == "test"]
    if not test.empty:
        print("\n=== TEST MASE by model x horizon (lower is better; <1 beats naive) ===")
        print(test.pivot(index="model", columns="h", values="mase").round(3).to_string())
    print(f"\nartifacts: {result.get('run_dir', '(not written)')}")


if __name__ == "__main__":
    main()
