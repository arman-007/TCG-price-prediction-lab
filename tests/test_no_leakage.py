"""Mutation test for temporal leakage: features at period <= t0 must be
bit-identical when everything AFTER t0 is rewritten. If a new feature ever
peeks forward, this fails."""

import numpy as np
import pandas as pd

from pricelab.features.tabular import add_targets, build_features

HORIZONS = [1, 3]


def make_panel(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2022-01-01", periods=36, freq="MS")
    frames = []
    for cid in range(1, 4):
        log_y = np.cumsum(rng.normal(0, 0.1, len(periods))) + 2.0 + cid
        frames.append(
            pd.DataFrame(
                {
                    "card_id": cid,
                    "period": periods,
                    "y": np.exp(log_y),
                    "log_y": log_y,
                    "is_observed": True,
                    "staleness": 0.0,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame({"card_id": [1, 2, 3], "genre": ["a", "a", "b"]})
    return panel, meta


def test_features_do_not_leak_future():
    panel, meta = make_panel()
    t0 = pd.Timestamp("2023-12-01")

    feat_a, cols = build_features(panel, meta)
    past_a = feat_a[feat_a["period"] <= t0].reset_index(drop=True)

    mutated = panel.copy()
    future = mutated["period"] > t0
    mutated.loc[future, "log_y"] += 5.0
    mutated.loc[future, "y"] = np.exp(mutated.loc[future, "log_y"])
    feat_b, _ = build_features(mutated, meta)
    past_b = feat_b[feat_b["period"] <= t0].reset_index(drop=True)

    for col in cols:
        a, b = past_a[col], past_b[col]
        if a.dtype.name == "category":
            assert (a == b).all(), col
        else:
            np.testing.assert_allclose(a.to_numpy("float64"), b.to_numpy("float64"),
                                       err_msg=col, equal_nan=True)


def test_targets_alignment():
    panel, meta = make_panel()
    feat, _ = build_features(panel, meta)
    feat = add_targets(feat, HORIZONS)
    one = feat[feat["card_id"] == 1].reset_index(drop=True)
    for h in HORIZONS:
        expected = one["log_y"].shift(-h) - one["log_y"]
        np.testing.assert_allclose(
            one[f"target_{h}"].to_numpy(), expected.to_numpy(), equal_nan=True
        )
        # last h rows have no target
        assert one[f"target_{h}"].iloc[-h:].isna().all()
        assert not one[f"target_observed_{h}"].iloc[-h:].any()
