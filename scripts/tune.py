#!/usr/bin/env python
"""Optuna-tune one model family; best params land in configs/tuned/<model>.yaml.

Examples:
    python scripts/tune.py --model prophet --trials 40
    python scripts/tune.py --model xgboost
"""

from pricelab.threadlimits import limit_blas_threads

limit_blas_threads(1)

import argparse
import logging

from pricelab.config import load_config
from pricelab.data.preprocess import build_panel
from pricelab.data.snapshot import load_snapshot
from pricelab.tuning.optuna_tuner import SEARCH_SPACES, tune_model
from pricelab.utils import set_seeds, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(SEARCH_SPACES))
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    setup_logging()
    logging.getLogger("optuna").setLevel(logging.WARNING)

    cfg = load_config(args.config, args.overrides)
    set_seeds(int(cfg.seed))
    prices, cards = load_snapshot(cfg.data.snapshot_dir)
    panel = build_panel(prices, cfg)

    result = tune_model(args.model, panel, cards, cfg, n_trials=args.trials, tag=args.tag)
    print(f"\nbest val MASE: {result['best_value']:.4f}")
    print(f"best params:   {result['best_params']}")
    print(f"written to:    {result['path']}")


if __name__ == "__main__":
    main()
