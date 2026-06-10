import numpy as np
import pandas as pd

from pricelab.evaluation.metrics import pinball_loss, qcol, summarize


def test_qcol():
    assert qcol(0.1) == "q10"
    assert qcol(0.5) == "q50"
    assert qcol(0.9) == "q90"


def test_pinball_known_values():
    actual = np.array([1.0, 2.0])
    pred = np.array([2.0, 1.0])
    # diffs (a-p) = [-1, +1]
    assert pinball_loss(actual, pred, 0.5) == 0.5
    assert np.isclose(pinball_loss(actual, pred, 0.1), (0.9 + 0.1) / 2)
    assert np.isclose(pinball_loss(actual, pred, 0.9), (0.1 + 0.9) / 2)


def test_summarize_basic():
    df = pd.DataFrame(
        {
            "model": "m",
            "split": "test",
            "h": 1,
            "actual_log": [1.0, 2.0, 3.0, 4.0],
            "pred_log": [1.5, 2.0, 2.0, 5.0],
            "last_log": [0.0, 0.0, 5.0, 5.0],
            "naive_scale": [0.5, 0.5, 0.5, 0.5],
            "q10": [0.0, 1.0, 2.5, 4.5],
            "q50": [1.5, 2.0, 2.0, 5.0],
            "q90": [2.0, 3.0, 3.5, 5.5],
            "fallback": [False, False, True, False],
            "fit_seconds": [0.1, 0.1, 0.1, 0.1],
        }
    )
    out = summarize(df, (0.1, 0.5, 0.9))
    assert len(out) == 1
    row = out.iloc[0]
    # |err| = [.5, 0, 1, 1] -> mase = mean/0.5 = 1.25
    assert np.isclose(row["mase"], 1.25)
    # coverage: actual within [q10,q90] for rows 1,2,3 -> 3/4
    assert np.isclose(row["coverage_80"], 0.75)
    # directions vs last: actual [+,+,-,-], pred [+,+,-,=0?] row3: pred 5-5=0 vs actual -1 -> wrong
    assert np.isclose(row["dir_acc"], 0.75)
    assert np.isclose(row["fallback_rate"], 0.25)
    assert row["n"] == 4
