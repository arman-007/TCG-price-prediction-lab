"""Model registry: config name -> forecaster.

Three families with different lifecycles, routed by the backtest harness:
  - LOCAL    fit per (card, cutoff) series
  - GLOBAL   fit once per cutoff on the pooled panel (xgboost)
  - POSTHOC  assembled from stored member forecasts (ensembles) — they never
    fit anything themselves, so they live in evaluation/backtest.py
"""

from __future__ import annotations

from .baselines import DriftForecaster, EtsForecaster, NaiveForecaster, SeasonalNaiveForecaster
from .monte_carlo import MonteCarloForecaster
from .prophet_model import ProphetForecaster
from .sarima import SarimaForecaster

LOCAL_MODELS = {
    NaiveForecaster.name: NaiveForecaster,
    DriftForecaster.name: DriftForecaster,
    SeasonalNaiveForecaster.name: SeasonalNaiveForecaster,
    EtsForecaster.name: EtsForecaster,
    SarimaForecaster.name: SarimaForecaster,
    ProphetForecaster.name: ProphetForecaster,
    MonteCarloForecaster.name: MonteCarloForecaster,
}

GLOBAL_MODELS = {"xgboost"}
POSTHOC_MODELS = {"ensemble", "ensemble_conformal"}


def make_local(name: str, cfg, quantiles, seed: int, freq_offset: str):
    """Instantiate a local forecaster with its config section (cfg.<name>)."""
    cls = LOCAL_MODELS[name]
    params = dict(cfg.get(name, {}) or {})
    if name == ProphetForecaster.name:
        return cls(params, quantiles=quantiles, seed=seed, freq=freq_offset)
    return cls(params, quantiles=quantiles, seed=seed)


def split_model_names(names: list[str]) -> tuple[list[str], list[str], list[str]]:
    local = [n for n in names if n in LOCAL_MODELS]
    global_ = [n for n in names if n in GLOBAL_MODELS]
    posthoc = [n for n in names if n in POSTHOC_MODELS]
    unknown = set(names) - set(local) - set(global_) - set(posthoc)
    if unknown:
        raise ValueError(f"unknown models in config: {sorted(unknown)}")
    return local, global_, posthoc
