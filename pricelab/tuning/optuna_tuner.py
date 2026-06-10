"""Optuna hyperparameter tuning.

Budget-aware design for a 12-core laptop:
  - tune on a deterministic subsample of the most-observed cards
    (tuning.max_cards) and a reduced cutoff count (tuning.n_cutoffs);
  - cutoffs are pushed back via backtest.exclude_last so they sit strictly
    inside the validation era of the main backtest — hyperparameters NEVER
    see the test windows;
  - objective: mean MASE across horizons (equal weight), TPE sampler,
    sqlite storage so studies resume across sessions.

Best params are written to configs/tuned/<model>.yaml in the same nested
shape as the main config, so `run_backtest.py --tuned` can merge them in.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import yaml

from ..config import Section, merge_into
from ..evaluation.backtest import run_backtest

logger = logging.getLogger(__name__)


def suggest_prophet(trial) -> dict:
    return {
        "prophet": {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.5, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.01, 10.0, log=True
            ),
            "n_changepoints": trial.suggest_int("n_changepoints", 5, 25),
            "changepoint_range": trial.suggest_float("changepoint_range", 0.7, 0.95),
            "yearly_seasonality": trial.suggest_categorical(
                "yearly_seasonality", [True, False]
            ),
        }
    }


def suggest_xgboost(trial) -> dict:
    return {
        "xgboost": {
            "params": {
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 20.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            }
        }
    }


def suggest_sarima(trial) -> dict:
    return {
        "sarima": {
            "max_p": trial.suggest_int("max_p", 1, 3),
            "max_q": trial.suggest_int("max_q", 1, 3),
            "d_options": trial.suggest_categorical(
                "d_options", ["0", "1", "01"]  # encoded; decoded below
            ),
            "seasonal": trial.suggest_categorical("seasonal", [False, True]),
        }
    }


def suggest_monte_carlo(trial) -> dict:
    return {
        "monte_carlo": {
            "block_size": trial.suggest_int("block_size", 1, 6),
            "drift": trial.suggest_categorical("drift", ["full", "half", "none"]),
        }
    }


def suggest_ets(trial) -> dict:
    return {
        "ets": {
            "trend": trial.suggest_categorical("trend", ["add", "none"]),
            "damped": trial.suggest_categorical("damped", [True, False]),
        }
    }


SEARCH_SPACES = {
    "prophet": suggest_prophet,
    "xgboost": suggest_xgboost,
    "sarima": suggest_sarima,
    "monte_carlo": suggest_monte_carlo,
    "ets": suggest_ets,
}

_D_DECODE = {"0": [0], "1": [1], "01": [0, 1]}


def _decode(overrides: dict) -> dict:
    sarima = overrides.get("sarima")
    if sarima and "d_options" in sarima:
        sarima["d_options"] = _D_DECODE[sarima["d_options"]]
    return overrides


def tune_model(
    model: str,
    panel,
    cards_meta,
    base_cfg: Section,
    n_trials: int | None = None,
    tag: str | None = None,
) -> dict:
    import optuna

    if model not in SEARCH_SPACES:
        raise ValueError(f"no search space for {model!r}; options: {sorted(SEARCH_SPACES)}")

    tcfg = base_cfg.get("tuning", Section()) or Section()
    n_trials = int(n_trials or tcfg.get("n_trials", 40))
    max_cards = int(tcfg.get("max_cards", 80))
    n_cutoffs = int(tcfg.get("n_cutoffs", 9))

    # Deterministic "most liquid first" subsample for cheap, stable trials.
    obs_count = (
        panel[panel["is_observed"]].groupby("card_id").size().sort_values(ascending=False)
    )
    keep = obs_count.index[:max_cards]
    sub_panel = panel[panel["card_id"].isin(keep)]

    # Keep tuning cutoffs out of the main backtest's test era.
    n_main = int(base_cfg.backtest.n_cutoffs)
    n_test_main = n_main - int(round(n_main * float(base_cfg.backtest.val_fraction)))

    def objective(trial):
        cfg = copy.deepcopy(base_cfg)
        merge_into(cfg, _decode(SEARCH_SPACES[model](trial)))
        cfg["models"] = [model]
        merge_into(
            cfg,
            {
                "backtest": {
                    "n_cutoffs": n_cutoffs,
                    "exclude_last": n_test_main,
                    "val_fraction": 0.0,  # every tuning cutoff scores
                }
            },
        )
        try:
            result = run_backtest(sub_panel, cards_meta, cfg, out_dir=None)
        except Exception as exc:
            logger.warning("trial %d failed: %s", trial.number, exc)
            return float("inf")
        metrics = result["metrics"]
        rows = metrics[metrics["model"] == model]
        if rows.empty or not np.isfinite(rows["mase"]).all():
            return float("inf")
        return float(rows["mase"].mean())  # equal weight across horizons

    storage_path = Path(tcfg.get("storage", "experiments/optuna.db"))
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{model}" + (f"-{tag}" if tag else ""),
        storage=f"sqlite:///{storage_path}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(base_cfg.seed)),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_overrides = _decode(_overrides_from_params(model, study.best_params))
    out_path = Path("configs/tuned") / f"{model}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(best_overrides, sort_keys=False))
    logger.info(
        "tuning %s done: best mase=%.4f after %d trials -> %s",
        model, study.best_value, len(study.trials), out_path,
    )
    return {"best_value": study.best_value, "best_params": study.best_params, "path": str(out_path)}


def _overrides_from_params(model: str, flat_params: dict) -> dict:
    """Rebuild the nested override dict from optuna's flat best_params."""
    if model == "xgboost":
        return {"xgboost": {"params": dict(flat_params)}}
    return {model: dict(flat_params)}
