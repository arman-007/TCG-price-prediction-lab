# TCG Price Prediction Lab — ML Engineer Handover

**Project:** `tcg-price-lab` (companion research workspace to the `jinvalex` Django backend)
**Status:** Track 1 complete and validated end to end · Track 2 scaffolded, blocked on data accumulation
**Last updated:** 2026-06-10 · all numbers below come from runs archived in `experiments/`

---

## 1. Executive summary

This lab answers one question for the jinvalex TCG platform: **which forecasting
model (or combination) actually predicts card prices better than a naive
random walk, measured honestly on held-out data?**

We benchmarked **SARIMA, Prophet, XGBoost, and Monte Carlo** (plus naive /
drift / ETS baselines and two ensembles) under a leakage-defended,
rolling-origin backtest with Optuna hyperparameter tuning.

**Headline result:** a **global XGBoost** model (one model pooled across all
cards, direct multi-horizon, multi-quantile) is the champion. It is the only
model at-or-better than naive at every horizon, the only one with real
directional signal (~62% direction accuracy at 6 months vs. coin-flip for
everything else), and its P10/P90 intervals are near-calibrated out of the
box. No ensemble combination beat it. Per-card classical models (SARIMA,
Prophet, ETS) **lose to naive** on this data — an important negative result,
because the app's current `predictions/` engine is built around exactly
those per-card models.

The realistic product framing this supports: *"direction + range over 3–6
months"* (e.g. "likely up, $53–$121"), not a confident point price.

---

## 2. Why this exists & how it aligns with jinvalex

The main repo has a `predictions/` app (SARIMA / Prophet / Monte Carlo
listed in `Prediction.MODEL_USED_CHOICES`) but no evidence about which
model is right for *this* data. Shipping forecasts without a backtest
harness risks confidently-wrong predictions in a paid product (FREE/PRO
tiers via RevenueCat).

Alignment points:

| jinvalex concept | lab counterpart |
|---|---|
| `cards.PriceHistory` (card, date, ungraded/graded prices, source) | snapshot → modeling panel |
| `predictions.Prediction` (`predicted_price`, `lower_bound`, `upper_bound`, `grade_target`, `model_used`, `date`) | `experiments/forecasts_latest.csv` columns map 1:1 (§10) |
| `grade_target` choices (ungraded / psa10 / bgs10) | `data.target_field` config (`ungraded_price`, `psa_10_price`, …) |
| product horizons "~30/90/180 days" | grid horizons 1/3/6 months |
| Celery-driven daily PriceCharting sync | the data regime Track 2 will be built on |

The lab is **deliberately separate** from the Django project (own folder,
own venv, own dependency pins — e.g. pandas 2.x here vs 3.x in the app) so
research can move fast without destabilizing the backend. The **only shared
surface is the database, accessed read-only** (§4).

---

## 3. The data — read this before modeling anything

```mermaid
flowchart LR
    PC["PriceCharting API"] -- "daily sync via celery<br/>since 2026-05-24" --> DJ["jinvalex Django app"]
    AP["Apify scraper"] -- "monthly backfill<br/>231 cards, 2020 to now" --> DJ
    DJ --> DB[("PostgreSQL<br/>cards_pricehistory: 1.36M rows<br/>cards_card: 599k rows")]
    DB -- "read-only snapshot<br/>(make snapshot)" --> RAW["data/raw parquet files (42 MB)"]
    subgraph LAB["tcg-price-lab (this repo)"]
        RAW --> PANEL["modeling panel<br/>(preprocess.py)"]
        PANEL --> BT["rolling-origin backtest"]
        BT --> EXPR["experiments/run dir:<br/>report.md, metrics.csv, forecasts.parquet"]
        PANEL --> FC["scripts/forecast.py"]
        FC --> OUT["forecasts_latest.csv<br/>(P10/P50/P90 per card)"]
    end
    OUT -. "graduation path (section 10)" .-> DJ
```

### 3.1 Two regimes (profiled 2026-06-10, see `reports/data_profile.md`)

| regime | cards | history | usable for |
|---|---|---|---|
| **Deep cohort** — Apify monthly backfill | **231** (~90% Pokémon) | monthly points, 2020/2022 → now; ~40 obs median | **Track 1 (done)**: per-card + global models on a monthly grid |
| **Everything else** — daily PriceCharting sync | ~599k | started **2026-05-24**; median **2 observations** | **Track 2 (future)**: global weekly model, viable ≈ Dec 2026 |

Key totals: 1,361,816 price rows · 599,431 cards · only **92 distinct dates**
(monthly first-of-month from the backfill + daily since 2026-05-24).

### 3.2 Data semantics that shaped the design

1. **Prices are standing estimates.** PriceCharting repeats a price until a
   sale moves it; many cards sit flat for months ($85.00 → $85.00 → …).
   Consequences:
   - forward-filling a bounded number of grid periods is *faithful* to the
     source (`ffill_limit: 2`), but evaluation must only score against
     **really observed** periods (`is_observed` flag) — never against our
     own forward-fills;
   - the MASE denominator must be floored (`MIN_MASE_SCALE = 1%/month`):
     measured here, rows with near-zero in-sample scale were **1.7% of
     forecasts but 96% of total scaled error** before the floor.
2. **Price levels span $1 → $2,500+** (panel median ≈ $11) → everything is
   modeled in **log space**; the point forecast is the log-median (robust
   for skewed prices). `exp(point)` is a median, not a mean — fine for our
   use.
3. **~40 monthly observations per card** → per-card ML is hopeless;
   per-card *statistical* models are viable; the GBM must be **global**
   (pooled across cards). That hypothesis won.
4. Apify and API sources agree in level at the switchover (verified — no
   level-jump artifact), but backfill values come from PriceCharting's
   *chart*, which is already smoothed → apparent volatility is understated
   in the backfill era.

Relevant tables (only these are read):

```mermaid
erDiagram
    cards_card ||--o{ cards_pricehistory : "price_history"
    cards_card {
        int id PK
        string pricecharting_id
        string product_name
        string console_name
        string genre
        int collection_id FK
    }
    cards_pricehistory {
        int card_id FK
        date date
        decimal ungraded_price
        decimal graded_price_columns "grade_7 ... sgc_10 (8 columns)"
        string source "apify or pricecharting_api"
    }
    cards_collection ||--o{ cards_card : "cards"
    cards_cardcategory ||--o{ cards_collection : "collections"
```

---

## 4. Database connection (the connector)

`pricelab/data/db.py` is the single DB touchpoint. Configuration lives in
`.env` at the repo root (`cp .env.example .env`), with **two supported
styles** — set ONE:

```bash
# Style 1 — single URL (takes precedence; handy for staging/remote)
DATABASE_URL=postgresql://jinvalex_user:***@localhost:5432/jinvalex_database

# Style 2 — individual vars (names match the main project's .env, copy verbatim)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=jinvalex_database
DB_USER=jinvalex_user
DB_PASSWORD=***
```

Guarantees and workflow:

- **Read-only is enforced at the connection level**
  (`options="-c default_transaction_read_only=on"`): an accidental
  INSERT/UPDATE/CREATE fails loudly inside Postgres. Verified live — a
  write probe returns `cannot execute CREATE TABLE in a read-only
  transaction`.
- The DB is hit **once per snapshot refresh** (`make snapshot` →
  `data/raw/*.parquet` + `manifest.json` with row counts/date range).
  Every experiment then runs off the immutable snapshot → reproducible
  results, zero load on the app DB during experimentation.
- Docker note: the compose `tcg_db` container does not publish 5432 to the
  host; the local Postgres at `localhost:5432` is the one the Django dev
  setup (and this lab) uses. To point at a different environment, change
  `DATABASE_URL` — nothing else.

---

## 5. Repository architecture

```
tcg-price-lab/
├── configs/
│   ├── default.yaml          # Track 1: monthly grid, deep cohort — THE config
│   ├── weekly_all.yaml       # Track 2 stub (do not expect results yet — §11)
│   └── tuned/*.yaml          # Optuna output, merged via --tuned
├── pricelab/
│   ├── config.py             # YAML -> attr-dict, dotted overrides, deep merge
│   ├── threadlimits.py       # BLAS pinning (import FIRST in scripts — numpy-free)
│   ├── data/
│   │   ├── db.py             # read-only Postgres connector (§4)
│   │   ├── snapshot.py       # DB -> parquet + manifest
│   │   └── preprocess.py     # snapshot -> panel (grid, ffill, staleness, filters)
│   ├── features/tabular.py   # leakage-safe GBM features (mutation-tested)
│   ├── models/
│   │   ├── base.py           # ForecastResult (log-space mean + quantile paths)
│   │   ├── baselines.py      # naive / drift / seasonal-naive / ETS
│   │   ├── sarima.py         # SARIMAX, small AIC grid, NaN-tolerant (Kalman)
│   │   ├── prophet_model.py  # log scale, MAP, quantiles via predictive_samples
│   │   ├── monte_carlo.py    # moving-block bootstrap of log returns
│   │   ├── xgboost_model.py  # GLOBAL direct multi-horizon multi-quantile GBM
│   │   └── registry.py       # name -> family routing (local / global / posthoc)
│   ├── evaluation/
│   │   ├── metrics.py        # MASE(+floor), sMAPE, pinball, coverage, dir-acc
│   │   └── backtest.py       # rolling-origin engine + ensembles + reports
│   └── tuning/optuna_tuner.py# search spaces, val-era-only cutoffs, sqlite studies
├── scripts/                  # make_snapshot · profile_data · run_backtest · tune · forecast
├── tests/                    # 11 tests incl. the leakage mutation test
├── experiments/              # one dir per run: config.yaml, forecasts.parquet,
│                             #   metrics.csv, report.md, weights.json  (gitignored)
└── reports/                  # data_profile.md + figures
```

Model lifecycles (how the backtest routes them):

```mermaid
flowchart TB
    subgraph PER_CUTOFF["for each cutoff C — only data up to C is visible"]
        direction TB
        subgraph LOCAL["LOCAL pass (joblib, 1 task per card)"]
            L1["naive, drift, ETS, SARIMA, Prophet, MonteCarlo<br/>fit per card+cutoff on log-price series"]
        end
        subgraph GLOBAL["GLOBAL pass (main process)"]
            G1["build leakage-safe features once (causal)"] --> G2["XGBoost per horizon h:<br/>train on rows with t + h up to C<br/>and really-observed targets only"]
        end
    end
    LOCAL --> F["forecast store<br/>(log mean + q10/q50/q90 per card x cutoff x h)"]
    GLOBAL --> F
    F --> J["join actuals — score ONLY rows whose<br/>target period was really observed"]
    J --> W["fit ensemble weights on VAL cutoffs<br/>(weight proportional to 1/MASE^2, per horizon)"]
    W --> E["posthoc: ensemble (Vincentized quantile blend)<br/>+ ensemble_conformal (split-conformal intervals)"]
    E --> R["metrics.csv, report.md, weights.json"]
```

---

## 6. Evaluation methodology (the part that keeps us honest)

**Rolling-origin backtest** on the monthly grid, horizons **1 / 3 / 6
months** (≈30/90/180 days):

```
periods:   ... 2024-06 ... 2025-02 | 2025-03 ... 2025-11 | 2025-12 2026-01 ... 2026-06
cutoffs:       c1 ... c9 (VAL)       c10 ... c18 (TEST)     <- last 6 reserved for actuals
                 |                      |
                 |                      +-- headline numbers; NEVER used for tuning
                 +-- hyperparameter tuning + ensemble-weight fitting
at each cutoff: every model sees data <= cutoff only; forecasts 1/3/6 steps ahead
```

- **Chronological val/test split of cutoffs** (`val_fraction: 0.5`). The
  Optuna tuner additionally sets `backtest.exclude_last` so its cutoffs sit
  strictly inside the val era — hyperparameters can never see test windows.
- **Metrics** (`metrics.py`): MASE in log space (primary; <1.0 beats the
  naive random walk; floored denominator — §3.2), median MASE (robust
  view), sMAPE/$-MAE (product intuition), directional accuracy (excluding
  no-move rows), pinball loss + 80%-interval coverage (probabilistic
  quality). NOTE: MASE grows with horizon mechanically (denominator is the
  1-step scale) — compare models *within* a horizon, never across.
- **Leakage defenses** (all enforced, most tested):
  - `tests/test_no_leakage.py` is a *mutation test*: it rewrites all data
    after t₀ and asserts every feature at ≤ t₀ is bit-identical;
  - GBM trains only on **really-observed** targets (no forward-fill echo →
    no "no-change" bias);
  - SARIMA's order is selected at a card's *first* cutoff only (a subset of
    later windows — legitimate) and reused;
  - any local-model failure falls back to naive **and is counted**
    (`fallback_rate` column — this is how we caught ETS silently failing
    100% of the time, see §9).
- **Determinism**: fixed seeds; per-card RNG via `stable_card_seed`
  (scheduling-independent across joblib workers); verified by
  `test_backtest_is_deterministic`.

---

## 7. Models — what, why, and tuned outcome

| model | family | design notes | Optuna outcome (val) |
|---|---|---|---|
| naive | local | last log price, √h Gaussian intervals. **The bar.** | — |
| drift / seasonal-naive | local | sanity baselines; drift damping knob | — |
| ETS | local | damped-trend exponential smoothing (statsmodels `ETSModel`), model-based intervals | chose `trend='none'` → collapses toward smoothed naive: *no trend structure to exploit* |
| SARIMA | local | `SARIMAX` with small AIC grid (p,q ≤ 2; d ∈ {0,1}; trend n/c); Kalman handles NaN gaps natively; **pmdarima deliberately avoided** (numpy-lag, overkill) | chose p,q ≤ 1, d=1, **seasonal(12) on** — helped, still < naive |
| Prophet | local | log scale, MAP, weekly/daily seasonality off, quantiles from `predictive_samples` | `yearly_seasonality=False`, `changepoint_range=0.70` (freeze last 30% — stabilizes extrapolation); −30% MASE vs defaults, **still worst** — wrong tool for standing prices |
| Monte Carlo | local | **moving-block bootstrap** of historical log returns (fat-tail honest, vs Gaussian GBM); per-step return clip; drift shrink knob | `block_size=2`, **`drift='full'`** — validation rewards keeping momentum |
| **XGBoost** | **global** | one model per horizon (direct, no recursive error build-up), `reg:quantileerror` with `quantile_alpha=[.1,.5,.9]` (one fit → calibrated bands), features: return lags, rolling vol, momentum, range position, staleness, calendar, genre + same-period genre median return | depth 2, 100 trees, lr 0.016 — heavy regularization; ≈ defaults (already in the right regime) |
| ensemble | posthoc | per-horizon weights ∝ 1/val-MASE², quantile averaging (Vincentization) | weights correctly concentrate on XGBoost |
| ensemble_conformal | posthoc | same blend; intervals replaced by split-conformal band from val residuals | fixed over-coverage: 83–89% → 79–81% at nominal 80% |

---

## 8. Results (test split — 9 cutoffs the tuner never saw)

Tuned full bake-off, 168 cards × 18 cutoffs (`experiments/20260610-134232-full-tuned/`):

| model | h=1 | h=3 | h=6 | dir-acc h=6 | coverage80 h=6 |
|---|---|---|---|---|---|
| **xgboost** | **1.314** | **3.288** | **4.870** | **0.62** | 0.79 |
| naive | 1.314 | 3.291 | 4.923 | — | 0.88 |
| monte_carlo | 1.321 | 3.321 | 5.056 | 0.51 | 0.80 |
| ets | 1.433 | 3.375 | 4.966 | 0.50 | 0.78 |
| ensemble | 1.441 | 3.439 | 5.098 | 0.57 | 0.83 |
| sarima | 1.540 | 3.617 | 5.351 | 0.25 | 0.76 |
| prophet | 2.676 | 4.705 | 6.693 | 0.51 | 0.45 |

(Untuned reference run: `experiments/20260610-131651-full/` — tuning improved
Prophet −30%, ETS −19%, SARIMA −10%, MC −5%; XGBoost moved within noise of
its defaults, and its *untuned* h=6 score was 4.78.)

**Ensemble subset experiments** (recomputed post-hoc from stored forecasts —
no refits needed): `xgb+naive` 1.314/3.289/4.894, `xgb+mc` 1.317/3.299/4.933,
`xgb+mc+ets` 1.345/3.308/4.916 → **no combination beats XGBoost alone**.

How to read this honestly:

- The **edge over naive is real but modest** (≈0.1% at h=3, 1–3% at h=6 on
  mean MASE) — 1-month prices are a random walk.
- The **strong signals are direction (62% at h=6) and calibrated ranges**,
  which is what the product should sell.
- Naive's "coverage 0.88" over-covers because its intervals are wide;
  XGBoost achieves ~nominal coverage with *tighter* bands.

---

## 9. What is COMPLETED ✅

- [x] Data profiling + regime analysis (`reports/data_profile.md`, §3)
- [x] Read-only DB connector with `DATABASE_URL` *and* `DB_*` styles, write-probe verified (§4)
- [x] Snapshot layer (DB → parquet + manifest; 42 MB)
- [x] Panel preprocessing: monthly grid, bounded ffill + staleness, observed-flag, min-obs/min-price/max-cards filters
- [x] 7 models + 2 ensembles behind a uniform interface (§7)
- [x] Rolling-origin backtest engine with val/test cutoff split, per-card parallelism (joblib, 8 workers ≈ 5.5 min full run)
- [x] Metrics suite incl. flat-series-safe MASE; per-run artifacts (config, forecasts, metrics, report, weights)
- [x] Optuna tuning for 5 families, val-era-only, sqlite-persisted, results in `configs/tuned/`
- [x] Tuned bake-off + ensemble-subset study → **champion: global XGBoost** (§8)
- [x] Production-style forecaster: `scripts/forecast.py` → `forecasts_latest.csv` (504 rows = 168 cards × 3 horizons)
- [x] 11 tests incl. leakage mutation test + determinism test
- [x] Bugs found & fixed by the harness itself:
  - flat-series MASE blowup (1.7% of rows = 96% of error mass) → 1%/month scale floor;
  - ETS silently failing 100% (statsmodels needs pandas input) → caught via `fallback_rate`, fixed, retested 25/25.

## 9b. What is PENDING ⏳

- [ ] **Initial git commit** — repo is `git init`-ed but nothing committed yet (deliberate: waiting for your call on history start).
- [ ] **Volatility segmentation in reports** — split metrics by flat/mobile cards to show *where* ML adds value (small change in `metrics.py`; highest-value next analysis).
- [ ] **Django integration** (§10) — nothing in the app consumes lab output yet; `Prediction.MODEL_USED_CHOICES` lacks `xgboost`/`ensemble` (needs a tiny migration).
- [ ] **Track 2** — global weekly model over all ~599k cards; *blocked on data* until the daily sync accumulates ~6 months (≈ 2026-12). `configs/weekly_all.yaml` is the prepared entry point; `features/tabular.py` must be vectorized first (notes inline in the module).
- [ ] Graded targets — config supports `psa_10_price` etc. (`data.target_field`) but only `ungraded_price` has been validated; PSA-10 coverage is ~44% of backfill rows, expect a smaller panel.
- [ ] Snapshot/retune cadence — no schedule yet; suggestion: monthly `make snapshot && make backtest-tuned`, retune quarterly.
- [ ] Cosmetics: statsmodels convergence warnings leak from workers (harmless — failures already fall back + are counted); no CI runner for tests.

## 9c. Improvements & future ideas 🔭 (prioritized)

1. **Sell direction + range, not points** (product framing; supported by the numbers today).
2. Volatility/liquidity segmentation (pending above) — likely shows XGBoost's edge concentrates in mobile cards.
3. Better features when obtainable: sales volume / `market_volume`, PSA population reports, set-rotation & reprint calendar, grading spread (PSA10/raw ratio), eBay sold-listing counts.
4. Model upgrades worth trying *after* Track 2 data lands: LightGBM comparison, hierarchical pooling (genre→card shrinkage), quantile calibration per volatility tier, statsforecast/darts baselines for an external sanity check.
5. Regime awareness: none of these models see reprints/bans/tournament results — an event-flag feature or simple changepoint monitor on residuals would catch breaks earlier.
6. Drift monitoring in production: track live MASE of shipped forecasts vs naive; alert if the edge disappears.

---

## 10. Graduation path into jinvalex

When you're ready to ship Track 1 output into the app:

1. **Schema**: `forecasts_latest.csv` → `predictions.Prediction`
   - `pred_price` → `predicted_price`, `p10_price` → `lower_bound`, `p90_price` → `upper_bound`
   - `target_period` → `date`, `model` → `model_used` (**add `xgboost` /
     `ensemble` to `MODEL_USED_CHOICES` — migration needed**)
   - `data.target_field` ↔ `grade_target` (`ungraded_price` ↔ `'ungraded'`).
2. **Mechanics** (two options, both fine):
   - *Loose coupling (recommended first)*: a Django management command /
     Celery beat task that runs `make snapshot && python scripts/forecast.py
     --tuned --weights <latest-run>/weights.json` in the lab and bulk-upserts
     the CSV. Zero new app dependencies.
   - *Tight coupling (later)*: port `GlobalXGBoost` + `features/tabular.py`
     into `predictions/engine/` once the champion is stable for a few
     monthly cycles.
3. **Scope honestly**: today's forecasts cover the 168-card deep cohort.
   Expansion to the full catalog is exactly Track 2.

---

## 11. Track 2 plan (the ~599k-card catalog)

- **Trigger**: ≈ 2026-12, when the daily sync (started 2026-05-24) gives
  ~26 weekly points per card.
- **Setup**: `configs/weekly_all.yaml` — weekly grid (`W-SUN`), horizons
  1/4/13 weeks, models naive/drift/MC/XGBoost (per-card SARIMA/Prophet are
  pointless at this depth and won't scale anyway).
- **Engineering prerequisite**: vectorize `features/tabular.py`
  (`groupby.transform(rolling)` lambdas are fine for 200 cards, not 600k —
  switch to `groupby.rolling` / pivot ops; notes are in the module docstring).
- **Expectation**: the global model becomes *more* powerful with 600k
  cross-sectional series — this is where the pooled approach really pays.

---

## 12. Runbook

```bash
# one-time
make setup            # python3.12 venv + pinned deps (incl. -e install)
cp .env.example .env  # set DATABASE_URL *or* DB_* (read-only enforced anyway)

# regular cycle
make snapshot         # refresh data/raw from DB (only step that touches the DB)
make profile          # -> reports/data_profile.md (rerun after every snapshot)
make test             # 11 tests, ~1 min
make backtest-fast    # ~30 s smoke (60 cards, 6 cutoffs)
make backtest         # full bake-off, ~6 min -> experiments/<ts>-full/report.md
make tune-prophet     # optuna (also: tune-xgboost / tune-sarima / tune-ets / tune-monte_carlo)
make backtest-tuned   # bake-off with configs/tuned/* merged
make forecast         # current forecasts -> experiments/forecasts_latest.{csv,parquet}

# escape hatches
venv/bin/python scripts/run_backtest.py --models naive,xgboost --set backtest.n_cutoffs=12 --tag myexp
venv/bin/python scripts/forecast.py --tuned --weights experiments/<run>/weights.json --champion xgboost
```

Conventions: every run writes a self-contained `experiments/<timestamp>-<tag>/`
with the exact config used; treat `report.md` files as the experiment log.
Compare runs only on the same snapshot. `MASE < 1.0` = beats naive; headline
numbers are TEST-split only.

### Glossary

- **MASE** — mean absolute error scaled by the card's in-sample mean 1-step
  move; scale-free across $1…$2,500 cards. Floored at 1%/month here (§3.2).
- **Rolling-origin** — refit at many historical cutoffs, forecast forward,
  score on what actually happened next.
- **Direct multi-horizon** — a separate model per horizon (no recursive
  feedback of own predictions).
- **Vincentization** — averaging member *quantiles* to blend probabilistic
  forecasts.
- **Split-conformal** — set interval width from the empirical quantile of
  validation residuals → distribution-free coverage guarantee.
- **Standing price** — PriceCharting's price persists until a sale moves it
  (why naive is so strong and why we only score real observations).
