# tcg-price-lab

> **New here? Start with [HANDOVER.md](HANDOVER.md)** — the full ML-engineer
> handover: context, architecture diagrams, methodology, results,
> completed/pending work, and the integration plan with the main app.

Standalone research workspace for **TCG card price forecasting** (Prophet /
SARIMA / XGBoost / Monte Carlo + ensembles). Deliberately separated from the
main Django project — the only shared surface is the Postgres database,
accessed **read-only** (`default_transaction_read_only=on`) and snapshotted
to local parquet so experiments never load the app DB.

The goal: find the model (or combination) that actually beats a naive
random walk on held-out data, with honest uncertainty intervals, before
anything ships into the app's `predictions/` engine.

## What the data allows (June 2026)

Profiling the DB (see `reports/data_profile.md`, regenerate with
`make profile`) shows **two regimes**:

| regime | cards | history | modeling status |
|---|---|---|---|
| deep cohort (Apify monthly backfill) | 231 (≈90% Pokémon) | monthly, 2020/2022 → now | **Track 1 — modeled now** |
| everything else (daily sync since 2026-05-24) | ~599k | median **2 observations** | **Track 2 — wait for history** |

Other realities that shaped the design:

- Prices are **standing estimates**: PriceCharting repeats a price until a
  sale moves it. Many series are flat for months. Hence: forward-fill is
  semantically valid (bounded by `ffill_limit`), but evaluation only scores
  against *real* observations, and the MASE denominator is floored
  (`MIN_MASE_SCALE = 1%/month`) so flat series can't explode the metric —
  measured here, 1.7% of rows otherwise contributed 96% of total ASE.
- Price levels span $1 → $2,500+ → **everything is modeled in log space**;
  point forecasts are log-medians (robust for skewed prices).
- Typical card: ~40 monthly points. Per-card ML is hopeless; per-card
  *statistical* models are viable; the GBM is **global** (pooled across
  cards) — that's the main hypothesis this lab tests.

## Model lineup

| model | type | role |
|---|---|---|
| naive / drift / seasonal-naive | local | the bar to beat (random walk) |
| ETS (damped trend) | local | strong classical baseline |
| SARIMA (AIC grid, statsmodels) | local | classical: trend + autocorrelation |
| Prophet (log scale, MAP) | local | trend changepoints, irregular-sampling-tolerant |
| Monte Carlo (block bootstrap of log returns) | local | fat-tail-honest probabilistic paths; drift shrinkage knob |
| XGBoost (global, direct multi-horizon, multi-quantile pinball) | global | cross-card regularities: mean reversion, genre momentum |
| ensemble | post-hoc | per-horizon inverse-val-MASE² weighted blend (means + quantiles) |
| ensemble_conformal | post-hoc | same blend, intervals recalibrated split-conformal on val residuals |

## Evaluation protocol (the part that keeps us honest)

- **Rolling-origin backtest**: 18 monthly cutoffs; at each cutoff every
  model sees only data ≤ cutoff; horizons 1/3/6 months (≈30/90/180 days).
- **Chronological val/test split of cutoffs**: earlier half = validation
  (hyperparameter tuning, ensemble weights), later half = **test** —
  headline numbers. The tuner additionally pushes its cutoffs back
  (`backtest.exclude_last`) so trials never touch test windows.
- **Metrics**: MASE (log space, primary — <1.0 beats naive), median MASE,
  sMAPE/$MAE for product intuition, directional accuracy, pinball loss and
  80%-interval coverage for the probabilistic side.
- **Leakage defenses**: features are mutation-tested (`tests/test_no_leakage.py`
  rewrites the future and asserts past features are bit-identical); GBM
  trains only on really-observed targets; SARIMA order chosen at a card's
  first cutoff only (subset of later windows); per-model failures fall back
  to naive and are *counted* (`fallback_rate` in reports).

## Repository guide

```
configs/            default.yaml (Track 1) · weekly_all.yaml (Track 2 stub) · tuned/ (optuna output)
pricelab/           the package (see table below)
scripts/            CLI entry points (snapshot, profile, backtest, tune, forecast)
tests/              metrics, preprocess, leakage mutation test, e2e smoke
reports/            data_profile.md + figures (committed — research log)
data/, experiments/ generated artifacts (gitignored, reproducible)
```

| file | what it does |
|---|---|
| `configs/default.yaml` | the Track 1 experiment config: cohort/grid/horizons, per-model params, backtest + tuning budgets. Every knob lives here. |
| `configs/weekly_all.yaml` | Track 2 stub (full catalog, weekly grid) — blocked on data until ~2026-12, see header comment |
| `configs/tuned/*.yaml` | Optuna best params per model; merged into runs via `--tuned` |
| `pricelab/config.py` | YAML → attr-dict config with dotted `--set key=value` overrides and deep merge |
| `pricelab/threadlimits.py` | BLAS thread pinning; numpy-free so scripts can import it *first* |
| `pricelab/utils/__init__.py` | logging setup, seeding, deterministic per-card RNG seeds |
| `pricelab/data/db.py` | **the DB connector**: read-only Postgres access via `DATABASE_URL` *or* `DB_*` vars |
| `pricelab/data/snapshot.py` | DB → `data/raw/*.parquet` + manifest; experiments never touch the DB directly |
| `pricelab/data/preprocess.py` | snapshot → modeling panel: monthly grid, bounded ffill + staleness, `is_observed` flag, quality filters |
| `pricelab/features/tabular.py` | leakage-safe features for the global GBM (return lags, rolling vol, momentum, calendar, genre cross-section) |
| `pricelab/models/base.py` | `ForecastResult` (log-space mean + quantile paths) and the local-forecaster interface |
| `pricelab/models/baselines.py` | naive / drift / seasonal-naive / ETS — the bar every model must beat |
| `pricelab/models/sarima.py` | SARIMAX with a small AIC-selected order grid; NaN-tolerant via Kalman filter |
| `pricelab/models/prophet_model.py` | Prophet on log prices, MAP, quantiles from predictive samples |
| `pricelab/models/monte_carlo.py` | moving-block bootstrap of log returns; drift-shrinkage knob |
| `pricelab/models/xgboost_model.py` | the champion: GLOBAL direct multi-horizon multi-quantile GBM pooled across cards |
| `pricelab/models/registry.py` | model-name → family routing (local / global / post-hoc) |
| `pricelab/evaluation/metrics.py` | MASE (flat-series-safe floor), sMAPE, pinball, coverage, directional accuracy |
| `pricelab/evaluation/backtest.py` | the rolling-origin engine: cutoffs, val/test split, parallel passes, ensembles, conformal intervals, report writing |
| `pricelab/tuning/optuna_tuner.py` | per-model search spaces; tuning cutoffs confined to the val era; sqlite-persisted studies |
| `scripts/make_snapshot.py` | pull a fresh read-only snapshot (`make snapshot`) |
| `scripts/profile_data.py` | regenerate `reports/data_profile.md` (`make profile`) |
| `scripts/run_backtest.py` | the bake-off CLI: `--fast`, `--tuned`, `--models`, `--set k=v` |
| `scripts/tune.py` | Optuna tuning CLI → `configs/tuned/<model>.yaml` |
| `scripts/forecast.py` | production-style current forecasts → `experiments/forecasts_latest.{csv,parquet}` |
| `tests/test_no_leakage.py` | the most important test: mutates the future, asserts past features are bit-identical |
| `HANDOVER.md` | full ML-engineer handover: context, diagrams, methodology, results, pending work |
| `Makefile` / `requirements.txt` / `pyproject.toml` | task runner · pinned deps · package metadata (`pip install -e .`) |
| `.env.example` | DB connection template — copy to `.env` (gitignored) |

## Setup — from clone to first forecast

Prerequisites: **Python 3.12**, a reachable **PostgreSQL with the jinvalex
database** (the main project's local Postgres or Docker `tcg_db`), ~1 GB
free disk for the venv, Linux/macOS.

```bash
# 1. clone
git clone git@github.com:arman-007/TCG-price-prediction-lab.git
cd TCG-price-prediction-lab

# 2. environment (~2 min, ~800 MB): venv + pinned deps + editable install
make setup

# 3. DB credentials — pick ONE style inside .env (connector is read-only either way)
cp .env.example .env
#    DATABASE_URL=postgresql://user:pass@localhost:5432/jinvalex_database
#    ...or fill the DB_* vars (same names as the main project's .env)

# 4. sanity check the code (no DB needed) — expect "11 passed"
make test

# 5. pull the data snapshot (~30 s; the ONLY step that touches the DB)
make snapshot         # -> data/raw/*.parquet + manifest.json

# 6. profile the snapshot — read this before modeling anything
make profile          # -> reports/data_profile.md

# 7. smoke-run the whole pipeline (~30 s: 60 cards, 6 cutoffs)
make backtest-fast    # prints a TEST MASE table; <1.0 beats naive

# 8. full bake-off (~6 min: 168 cards, 18 cutoffs, all models)
make backtest         # -> experiments/<timestamp>-full/report.md

# 9. (optional) hyperparameter tuning, ~5-10 min per family
make tune-xgboost
make tune-prophet     # also: tune-sarima / tune-ets / tune-monte_carlo

# 10. bake-off with tuned params merged
make backtest-tuned   # -> experiments/<timestamp>-full-tuned/

# 11. current forecasts for the app (P10/P50/P90 per card x horizon)
venv/bin/python scripts/forecast.py --tuned \
    --weights experiments/<your-tuned-run>/weights.json --champion xgboost
#   -> experiments/forecasts_latest.csv  (504 rows = 168 cards x 3 horizons)
```

Routine afterwards: `make snapshot && make backtest-tuned` monthly,
re-tune quarterly. Every run writes a self-contained directory under
`experiments/` with its exact config — results are reproducible from the
same snapshot (fixed seeds, deterministic per-card RNG). Treat the
`report.md` files as the experiment log; headline numbers are TEST-split only.

## Findings so far

See `experiments/*/report.md` for full tables. From the tuned full bake-off
(168 cards, 18 monthly cutoffs, 9 held-out test cutoffs — June 2026):

TEST MASE (lower = better, naive = the bar):

| model | h=1 | h=3 | h=6 | notes |
|---|---|---|---|---|
| **xgboost (global)** | **1.314** | **3.288** | **4.870** | only model ≥ naive everywhere; dir-acc 0.56–0.62; coverage ≈ nominal |
| naive | 1.314 | 3.291 | 4.923 | the bar |
| monte_carlo (tuned) | 1.321 | 3.321 | 5.056 | best probabilistic baseline; tuner kept full drift (momentum is real) |
| ets (tuned) | 1.433 | 3.375 | 4.966 | tuner chose trend='none' — collapses toward smoothed naive |
| sarima (tuned) | 1.540 | 3.617 | 5.351 | seasonal(12) helped slightly; still < naive |
| prophet (tuned) | 2.676 | 4.705 | 6.693 | −30% vs defaults but still worst; wrong tool for standing prices |

- **Champion: global XGBoost.** Post-hoc member-subset experiments (cheap —
  ensembles recompute from stored forecasts) showed *no* blend beats it;
  xgb+naive merely matches it. Tuning moved XGBoost within noise of its
  (already well-regularized) defaults — depth 2–3, ≤300 trees, lr ≤0.05.
- Short horizon (1m) is essentially unpredictable (random-walk standing
  prices); the exploitable signal is at 3–6 months and in *direction*
  (62% directional accuracy at h=6 vs coin-flip for everything else).
- Split-conformal recalibration works as designed: raw ensemble intervals
  over-covered (83–89% at nominal 80%), conformal landed 79–81%.
- Tuning improved every family (Prophet −30%, ETS −19%, SARIMA −10%,
  MC −5% mean MASE) — `make tune-<model>` then `make backtest-tuned`.

## Roadmap

1. Tune Prophet + XGBoost (+ MC block/drift), re-run, lock a champion combo.
2. Volatility segmentation in reports: where does ML add value vs flat cards
   where "no change" is trivially right?
3. **Track 2** (when daily sync has ~6 months of history, ~2026-12): global
   weekly model over the full catalog — `configs/weekly_all.yaml` is the
   prepared entry point; vectorize `features/tabular.py` first (notes inline).
4. Richer features when available: sales volume, population reports, set
   rotation/reprint calendar, grading-spread (PSA10/raw ratio) signals.
5. Graduation path: champion config + `scripts/forecast.py` output schema
   maps 1:1 onto the app's `predictions.Prediction` model
   (`predicted_price`, `lower_bound`, `upper_bound`, `model_used`).

## Limitations to keep in mind

- The deep cohort is 231 cards and ~90% Pokémon — conclusions may not
  transfer to Magic/YuGiOh until Track 2 data matures.
- Apify backfill is scraped from PriceCharting's chart, which is already a
  smoothed estimate; very low apparent volatility is partly an artifact.
- No exogenous events (reprints, bans, tournament results) — regime breaks
  will be missed by all of these models.
