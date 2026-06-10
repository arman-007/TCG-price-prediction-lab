"""Rolling-origin backtest engine.

Protocol:
  - Cutoffs: the last `n_cutoffs` grid periods (stepping `step`) that still
    leave max(horizon) future periods for actuals.
  - Chronological split of cutoffs: the EARLIER `val_fraction` share is
    "val" (used for hyperparameter tuning and ensemble weight fitting); the
    later share is "test" (headline numbers). Never tune on test.
  - At each cutoff every model only sees data <= cutoff. Local models are
    refit per (card, cutoff); the global GBM is refit per cutoff on the
    pooled panel.
  - Forecasts are scored ONLY against periods where a real observation
    landed (is_observed) — never against our own forward-fills.
  - A local model that raises is replaced by a naive fallback and flagged;
    fallback_rate is reported so silent degradation is visible.

Parallelism: one joblib task per CARD (all cutoffs x local models inside),
which amortizes worker spin-up and lets SARIMA reuse the order selected at
the card's first cutoff (legitimate: that window is a subset of later ones).

Ensembles are post-hoc: weights fitted on val cutoffs per horizon
(inverse-MASE^power over member models), applied everywhere; a split-
conformal variant recalibrates the interval width from val residuals.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import Section, dump_config
from ..features.tabular import add_targets, build_features
from ..models.baselines import NaiveForecaster
from ..models.registry import make_local, split_model_names
from ..models.xgboost_model import GlobalXGBoost
from ..utils import stable_card_seed
from .metrics import MIN_MASE_SCALE, qcol, summarize

logger = logging.getLogger(__name__)


def period_freq_to_offset(freq: str) -> str:
    """Map a pandas *period* alias to an *offset* alias matching the grid
    timestamps (period starts). Monthly periods start at month starts; any
    weekly grid is just 7-day spacing from the last stamp."""
    if freq == "M":
        return "MS"
    if freq.startswith("W"):
        return "7D"
    return freq


def choose_cutoffs(
    periods: list[pd.Timestamp], horizons, n_cutoffs: int, step: int, exclude_last: int = 0
):
    """`exclude_last` drops the N most recent eligible cutoffs — the tuner
    uses it to keep its cutoffs strictly inside the validation era so
    hyperparameters never see the test windows."""
    max_h = max(horizons)
    eligible = periods[: len(periods) - max_h]
    if exclude_last > 0:
        eligible = eligible[: len(eligible) - exclude_last]
    if not eligible:
        raise ValueError("series too short for the requested horizons")
    idx = list(range(len(eligible) - 1, -1, -step))[:n_cutoffs]
    return [eligible[i] for i in sorted(idx)]


# --------------------------------------------------------------------------
# Local-model pass (one task per card)
# --------------------------------------------------------------------------

def _run_card(
    card_id: int,
    y: pd.Series,
    observed: pd.Series,
    cutoffs: list[pd.Timestamp],
    periods: list[pd.Timestamp],
    local_names: list[str],
    model_cfgs: dict,
    quantiles: tuple[float, ...],
    base_seed: int,
    min_train_obs: int,
    horizons: list[int],
    freq_offset: str,
) -> list[dict]:
    rows: list[dict] = []
    max_h = max(horizons)
    seed = stable_card_seed(base_seed, card_id)
    period_pos = {p: i for i, p in enumerate(periods)}
    sarima_fixed = None

    for cutoff in cutoffs:
        y_train = y[y.index <= cutoff]
        obs_train = y_train.dropna()
        n_obs = int(observed[observed.index <= cutoff].sum())
        if n_obs < min_train_obs or obs_train.empty:
            continue
        diffs = obs_train.diff().dropna().abs()
        naive_scale = (
            float(max(diffs.mean(), MIN_MASE_SCALE)) if len(diffs) else MIN_MASE_SCALE
        )
        last_log = float(obs_train.iloc[-1])
        ci = period_pos[cutoff]

        for name in local_names:
            t0 = perf_counter()
            fallback = False
            try:
                model = make_local(name, model_cfgs, quantiles, seed, freq_offset)
                if name == "sarima" and sarima_fixed is not None:
                    model.params["fixed_order"] = sarima_fixed
                model.fit(y_train)
                res = model.predict(max_h).sorted_quantiles()
                if name == "sarima" and sarima_fixed is None:
                    sarima_fixed = getattr(model, "selected", None)
            except Exception:
                res = NaiveForecaster({}, quantiles, seed).fit(y_train).predict(max_h)
                fallback = True
            seconds = perf_counter() - t0

            for h in horizons:
                ti = ci + h
                if ti >= len(periods):
                    continue
                row = {
                    "card_id": card_id,
                    "cutoff": cutoff,
                    "model": name,
                    "h": h,
                    "target_period": periods[ti],
                    "pred_log": float(res.mean[h - 1]),
                    "last_log": last_log,
                    "naive_scale": naive_scale,
                    "n_train_obs": n_obs,
                    "fallback": fallback,
                    "fit_seconds": seconds / len(horizons),
                }
                for q in quantiles:
                    row[qcol(q)] = float(res.quantiles[q][h - 1])
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Global-model pass (XGBoost, fit per cutoff in the main process)
# --------------------------------------------------------------------------

def _run_global_xgb(panel, cards_meta, cfg, cutoffs, periods, quantiles, horizons, card_stats):
    xcfg = cfg.get("xgboost", Section()) or Section()
    feat, fcols = build_features(
        panel,
        cards_meta,
        lags=tuple(xcfg.get("lags", [1, 2, 3, 6, 12])),
        windows=tuple(xcfg.get("rolling_windows", [3, 6, 12])),
    )
    feat = add_targets(feat, horizons)
    period_pos = {p: i for i, p in enumerate(periods)}
    feat["period_idx"] = feat["period"].map(period_pos)

    rows: list[dict] = []
    min_train_obs = int(cfg.backtest.min_train_obs)
    for cutoff in cutoffs:
        ci = period_pos[cutoff]
        t0 = perf_counter()
        try:
            gx = GlobalXGBoost(dict(xcfg), horizons, quantiles, seed=int(cfg.seed)).fit(
                feat, fcols, ci
            )
        except Exception as exc:  # not enough pooled rows at early cutoffs
            logger.warning("xgboost skipped at cutoff %s: %s", cutoff, exc)
            continue
        preds = gx.predict_at_cutoff(feat, ci)
        seconds = perf_counter() - t0
        per_pred = seconds / max(len(preds), 1) / len(horizons)

        for card_id, res in preds.items():
            stats = card_stats.get((card_id, cutoff))
            if stats is None or stats["n_train_obs"] < min_train_obs:
                continue  # same admission gate as the local models — fairness
            for h in horizons:
                if h not in gx.models:
                    continue
                ti = ci + h
                if ti >= len(periods):
                    continue
                row = {
                    "card_id": card_id,
                    "cutoff": cutoff,
                    "model": "xgboost",
                    "h": h,
                    "target_period": periods[ti],
                    "pred_log": float(res.mean[h - 1]),
                    "last_log": stats["last_log"],
                    "naive_scale": stats["naive_scale"],
                    "n_train_obs": stats["n_train_obs"],
                    "fallback": False,
                    "fit_seconds": per_pred,
                }
                for q in quantiles:
                    row[qcol(q)] = float(res.quantiles[q][h - 1])
                rows.append(row)
    return rows


def _compute_card_stats(panel, cutoffs) -> dict:
    """(card_id, cutoff) -> last_log / naive_scale / n_train_obs, so global
    and post-hoc models share the exact admission gate and MASE scale used
    by the local pass."""
    out: dict = {}
    for card_id, gdf in panel.groupby("card_id"):
        y = gdf.set_index("period")["log_y"].dropna()
        obs_periods = gdf.loc[gdf["is_observed"], "period"].to_numpy()
        for cutoff in cutoffs:
            yt = y[y.index <= cutoff]
            if yt.empty:
                continue
            diffs = yt.diff().dropna().abs()
            out[(card_id, cutoff)] = {
                "last_log": float(yt.iloc[-1]),
                "naive_scale": (
                    float(max(diffs.mean(), MIN_MASE_SCALE)) if len(diffs) else MIN_MASE_SCALE
                ),
                "n_train_obs": int((obs_periods <= np.datetime64(cutoff)).sum()),
            }
    return out


# --------------------------------------------------------------------------
# Post-hoc ensembles
# --------------------------------------------------------------------------

def fit_ensemble_weights(val_eval: pd.DataFrame, members: list[str], power: float) -> dict:
    """Per-horizon inverse-MASE^power weights from validation-cutoff errors."""
    mase = (
        (val_eval["actual_log"] - val_eval["pred_log"]).abs() / val_eval["naive_scale"]
    ).groupby([val_eval["model"], val_eval["h"]]).mean()
    weights: dict = {}
    for h in sorted(val_eval["h"].unique()):
        w = {}
        for m in members:
            if (m, h) in mase.index:
                w[m] = (1.0 / max(float(mase.loc[(m, h)]), 1e-6)) ** power
        total = sum(w.values())
        if total > 0:
            weights[int(h)] = {m: v / total for m, v in w.items()}
    return weights


def blend_forecasts(
    forecasts: pd.DataFrame, weights: dict, quantiles, model_name: str = "ensemble"
) -> pd.DataFrame:
    """Weighted blend of member forecasts: means AND quantile paths
    (quantile averaging, a.k.a. Vincentization). Weights renormalize over
    the members actually present per (card, cutoff, h) group."""
    mdf = forecasts.copy()
    mdf["w"] = [
        weights.get(int(h), {}).get(m, np.nan) for m, h in zip(mdf["model"], mdf["h"])
    ]
    mdf = mdf[mdf["w"].notna()]
    if mdf.empty:
        return pd.DataFrame()

    value_cols = ["pred_log"] + [qcol(q) for q in quantiles]
    keys = ["card_id", "cutoff", "h", "target_period"]
    carry = ["last_log", "naive_scale", "n_train_obs"]

    weighted = mdf[value_cols].multiply(mdf["w"], axis=0)
    weighted[keys] = mdf[keys]
    weighted["w"] = mdf["w"]
    agg = weighted.groupby(keys).sum(min_count=1)
    blended = agg[value_cols].div(agg["w"], axis=0).reset_index()

    meta = mdf.groupby(keys)[carry].first().reset_index()
    fit_sec = mdf.groupby(keys)["fit_seconds"].sum().reset_index()
    out = blended.merge(meta, on=keys).merge(fit_sec, on=keys)
    out["model"] = model_name
    out["fallback"] = False

    # Quantile crossing guard after averaging.
    qcols = [qcol(q) for q in sorted(quantiles)]
    out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    return out


def _ensemble_rows(forecasts, eval_df, cfg, quantiles) -> tuple[pd.DataFrame, dict]:
    ecfg = cfg.get("ensemble", Section()) or Section()
    requested = list(ecfg.get("members", ["ets", "sarima", "prophet", "xgboost"]))
    present = set(forecasts["model"].unique())
    members = [m for m in requested if m in present]
    if len(members) < 2:
        logger.warning("ensemble skipped: <2 members present (%s)", members)
        return pd.DataFrame(), {}

    val = eval_df[(eval_df["split"] == "val") & eval_df["model"].isin(members)]
    if val.empty:
        logger.warning("ensemble skipped: no val rows to fit weights")
        return pd.DataFrame(), {}

    weights = fit_ensemble_weights(val, members, float(ecfg.get("power", 2.0)))
    out = blend_forecasts(forecasts[forecasts["model"].isin(members)], weights, quantiles)
    return out, weights


def _conformalize(ens_rows, eval_ens_val, quantiles) -> pd.DataFrame:
    """Split-conformal recalibration: replace the blended interval with
    P50 +/- the (qmax-qmin) empirical quantile of |val residuals| per h."""
    if ens_rows.empty or eval_ens_val.empty:
        return pd.DataFrame()
    qmin, qmax = min(quantiles), max(quantiles)
    coverage = qmax - qmin
    out = ens_rows.copy()
    out["model"] = "ensemble_conformal"
    for h, grp in eval_ens_val.groupby("h"):
        resid = (grp["actual_log"] - grp["pred_log"]).abs()
        if len(resid) < 20:
            continue
        r = float(np.quantile(resid, coverage))
        mask = out["h"] == h
        out.loc[mask, qcol(qmin)] = out.loc[mask, "pred_log"] - r
        out.loc[mask, qcol(qmax)] = out.loc[mask, "pred_log"] + r
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_backtest(
    panel: pd.DataFrame,
    cards_meta: pd.DataFrame,
    cfg: Section,
    out_dir: str | Path | None = None,
    tag: str | None = None,
) -> dict:
    t_start = time.time()
    quantiles = tuple(float(q) for q in cfg.get("quantiles", [0.1, 0.5, 0.9]))
    horizons = [int(h) for h in cfg.horizons]
    local_names, global_names, posthoc_names = split_model_names(list(cfg.models))
    freq_offset = period_freq_to_offset(str(cfg.data.freq))

    periods = sorted(pd.Timestamp(p) for p in panel["period"].unique())
    cutoffs = choose_cutoffs(
        periods,
        horizons,
        int(cfg.backtest.n_cutoffs),
        int(cfg.backtest.step),
        int(cfg.backtest.get("exclude_last", 0)),
    )
    n_val = int(round(len(cutoffs) * float(cfg.backtest.val_fraction)))
    val_cutoffs = set(cutoffs[:n_val])
    logger.info(
        "backtest: %d cards, %d cutoffs (%s..%s), %d val / %d test, horizons=%s",
        panel["card_id"].nunique(), len(cutoffs), cutoffs[0].date(), cutoffs[-1].date(),
        n_val, len(cutoffs) - n_val, horizons,
    )

    model_cfgs = {name: dict(cfg.get(name, {}) or {}) for name in local_names}
    min_train_obs = int(cfg.backtest.min_train_obs)

    # ---- local pass ----
    rows: list[dict] = []
    if local_names:
        tasks = [
            (cid, g.set_index("period")["log_y"], g.set_index("period")["is_observed"])
            for cid, g in panel.groupby("card_id")
        ]
        results = Parallel(n_jobs=int(cfg.runtime.n_jobs), prefer="processes")(
            delayed(_run_card)(
                cid, y, obs, cutoffs, periods, local_names, model_cfgs,
                quantiles, int(cfg.seed), min_train_obs, horizons, freq_offset,
            )
            for cid, y, obs in tasks
        )
        for r in results:
            rows.extend(r)
        logger.info("local pass done: %d forecast rows", len(rows))

    # ---- global pass ----
    card_stats = _compute_card_stats(panel, cutoffs)
    if "xgboost" in global_names:
        xgb_rows = _run_global_xgb(
            panel, cards_meta, cfg, cutoffs, periods, quantiles, horizons, card_stats
        )
        rows.extend(xgb_rows)
        logger.info("global pass done: %d xgboost rows", len(xgb_rows))

    if not rows:
        raise RuntimeError("backtest produced no forecasts — check filters/config")
    forecasts = pd.DataFrame(rows)

    # ---- join actuals; honest evaluation only on really-observed targets ----
    actuals = panel[["card_id", "period", "log_y", "is_observed"]].rename(
        columns={"period": "target_period", "log_y": "actual_log", "is_observed": "target_observed"}
    )
    forecasts = forecasts.merge(actuals, on=["card_id", "target_period"], how="left")
    forecasts["target_observed"] = forecasts["target_observed"].fillna(False).astype(bool)
    forecasts["split"] = np.where(forecasts["cutoff"].isin(val_cutoffs), "val", "test")

    def evaluable(df):
        return df[df["target_observed"] & df["actual_log"].notna()]

    eval_df = evaluable(forecasts)

    # ---- post-hoc ensembles ----
    weights: dict = {}
    if "ensemble" in posthoc_names or "ensemble_conformal" in posthoc_names:
        ens_rows, weights = _ensemble_rows(forecasts, eval_df, cfg, quantiles)
        if not ens_rows.empty:
            ens_rows = ens_rows.merge(actuals, on=["card_id", "target_period"], how="left")
            ens_rows["target_observed"] = ens_rows["target_observed"].fillna(False).astype(bool)
            ens_rows["split"] = np.where(ens_rows["cutoff"].isin(val_cutoffs), "val", "test")
            extra = [ens_rows] if "ensemble" in posthoc_names else []
            if "ensemble_conformal" in posthoc_names:
                conf = _conformalize(
                    ens_rows, evaluable(ens_rows[ens_rows["split"] == "val"]), quantiles
                )
                if not conf.empty:
                    extra.append(conf)
            forecasts = pd.concat([forecasts, *extra], ignore_index=True)
            eval_df = evaluable(forecasts)

    metrics = summarize(eval_df, quantiles)

    # ---- artifacts ----
    artifacts: dict = {"metrics": metrics, "forecasts": forecasts, "weights": weights}
    if out_dir is not None:
        run_id = time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, run_dir / "config.yaml")
        forecasts.to_parquet(run_dir / "forecasts.parquet", index=False)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        (run_dir / "weights.json").write_text(json.dumps(weights, indent=2))
        _write_report(run_dir, cfg, metrics, forecasts, weights, cutoffs, n_val, time.time() - t_start)
        artifacts["run_dir"] = str(run_dir)
        logger.info("artifacts written to %s", run_dir)
    return artifacts


def _write_report(run_dir, cfg, metrics, forecasts, weights, cutoffs, n_val, elapsed):
    lines = [
        "# Backtest report",
        "",
        f"- run dir: `{run_dir.name}`",
        f"- cohort: {cfg.data.cohort} | target: {cfg.data.target_field} | grid: {cfg.data.freq}",
        f"- cards: {forecasts['card_id'].nunique()} | cutoffs: {len(cutoffs)} "
        f"({cutoffs[0].date()} .. {cutoffs[-1].date()}; first {n_val} = val) | "
        f"horizons: {list(cfg.horizons)}",
        f"- wall time: {elapsed:.1f}s",
        "",
        "MASE < 1.0 beats the naive random walk. Headline = TEST split.",
        "",
    ]
    for split in ("test", "val"):
        sub = metrics[metrics["split"] == split]
        if sub.empty:
            continue
        lines.append(f"## {split.upper()} — MASE by model x horizon")
        pivot = sub.pivot(index="model", columns="h", values="mase").round(3).sort_values(
            by=sub["h"].min(), axis=0
        )
        lines += [pivot.to_markdown(), ""]
        cov_cols = [c for c in sub.columns if c.startswith("coverage_")]
        keep = ["model", "h", "n", "mase", "mase_median", "smape_price", "dir_acc",
                *cov_cols, "fallback_rate", "fit_seconds"]
        lines.append(f"## {split.upper()} — full metrics")
        lines += [sub[keep].round(4).to_markdown(index=False), ""]
    if weights:
        lines.append("## Ensemble weights (fit on val)")
        lines += ["```json", json.dumps(weights, indent=2), "```", ""]
    (run_dir / "report.md").write_text("\n".join(lines))
