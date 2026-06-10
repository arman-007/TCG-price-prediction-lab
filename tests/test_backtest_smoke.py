"""End-to-end backtest on synthetic data: cheap models only (no SARIMA/
Prophet — covered by the real fast run), exercising the full plumbing:
cutoff selection, local + global passes, observed-target gating, ensembles,
artifact writing."""

import numpy as np
import pandas as pd

from pricelab.config import Section
from pricelab.data.preprocess import build_panel
from pricelab.evaluation.backtest import choose_cutoffs, run_backtest


def make_cfg() -> Section:
    return Section.wrap(
        {
            "seed": 7,
            "quantiles": [0.1, 0.5, 0.9],
            "horizons": [1, 3],
            "data": {
                "target_field": "ungraded_price",
                "cohort": "all",
                "freq": "M",
                "min_obs": 10,
                "ffill_limit": 2,
                "min_price": 1.0,
                "max_cards": None,
            },
            "backtest": {
                "n_cutoffs": 4,
                "step": 1,
                "exclude_last": 0,
                "val_fraction": 0.5,
                "min_train_obs": 8,
            },
            "models": ["naive", "drift", "ets", "monte_carlo", "xgboost",
                        "ensemble", "ensemble_conformal"],
            "ets": {},
            "monte_carlo": {"n_paths": 200, "block_size": 2, "drift": "half"},
            "xgboost": {"lags": [1, 2, 3], "rolling_windows": [3, 6],
                        "params": {"n_estimators": 30, "max_depth": 2}},
            "ensemble": {"members": ["naive", "drift", "monte_carlo"], "power": 2.0},
            "runtime": {"n_jobs": 1},
        }
    )


def make_prices(n_cards: int = 8, n_months: int = 36, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_months, freq="MS") + pd.Timedelta(days=9)
    rows = []
    for cid in range(1, n_cards + 1):
        log_p = np.cumsum(rng.normal(0.005, 0.08, n_months)) + np.log(20 + cid)
        for d, lp in zip(dates, log_p):
            rows.append(
                {"card_id": cid, "date": d, "ungraded_price": float(np.exp(lp)),
                 "source": "pricecharting_api"}
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_choose_cutoffs_respects_horizons_and_exclusion():
    periods = list(pd.date_range("2024-01-01", periods=12, freq="MS"))
    cut = choose_cutoffs(periods, horizons=[1, 3], n_cutoffs=4, step=1)
    assert cut[-1] == periods[8]  # leaves 3 periods of actuals
    assert len(cut) == 4
    cut2 = choose_cutoffs(periods, horizons=[1, 3], n_cutoffs=4, step=1, exclude_last=2)
    assert cut2[-1] == periods[6]


def test_backtest_end_to_end(tmp_path):
    cfg = make_cfg()
    panel = build_panel(make_prices(), cfg)
    result = run_backtest(panel, cards_meta=pd.DataFrame(
        {"card_id": panel["card_id"].unique(), "genre": "test"}
    ), cfg=cfg, out_dir=tmp_path, tag="smoke")

    metrics, forecasts = result["metrics"], result["forecasts"]
    assert not metrics.empty
    assert set(metrics["split"]) == {"val", "test"}
    for name in ("naive", "drift", "monte_carlo", "ensemble", "ensemble_conformal"):
        assert name in set(metrics["model"]), name
    assert np.isfinite(metrics["mase"]).all()
    assert metrics["coverage_80"].between(0, 1).all()

    # every evaluated row scored against a real observation
    scored = forecasts[forecasts["target_observed"]]
    assert scored["actual_log"].notna().all()

    run_dir = result["run_dir"]
    for artifact in ("metrics.csv", "forecasts.parquet", "report.md", "config.yaml"):
        assert (tmp_path / run_dir.split("/")[-1] / artifact).exists() or \
               (pd.io.common.os.path.exists(f"{run_dir}/{artifact}"))


def test_backtest_is_deterministic():
    cfg = make_cfg()
    cfg["models"] = ["naive", "monte_carlo"]
    panel = build_panel(make_prices(), cfg)
    meta = pd.DataFrame({"card_id": panel["card_id"].unique(), "genre": "test"})
    a = run_backtest(panel, meta, cfg, out_dir=None)["forecasts"]
    b = run_backtest(panel, meta, cfg, out_dir=None)["forecasts"]

    def normalize(df):
        return (
            df.drop(columns=["fit_seconds"])  # wall-clock, not meant to be stable
            .sort_values(["model", "card_id", "cutoff", "h"])
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(normalize(a), normalize(b))
