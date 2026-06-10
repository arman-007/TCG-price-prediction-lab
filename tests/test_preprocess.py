import numpy as np
import pandas as pd

from pricelab.config import Section
from pricelab.data.preprocess import build_panel


def make_cfg(**data_over) -> Section:
    return Section.wrap(
        {
            "data": {
                "target_field": "ungraded_price",
                "cohort": "all",
                "freq": "M",
                "min_obs": 10,
                "ffill_limit": 2,
                "min_price": 1.0,
                "max_cards": None,
                **data_over,
            }
        }
    )


def make_prices() -> pd.DataFrame:
    # 14 mid-month observation dates, one per calendar month
    months = pd.date_range("2024-01-01", periods=14, freq="MS") + pd.Timedelta(days=14)
    rows = []
    # card 1: 14 monthly obs with a 4-month hole (indices 5..8)
    for i, d in enumerate(months):
        if 5 <= i <= 8:
            continue
        rows.append({"card_id": 1, "date": d, "ungraded_price": 10.0 + i, "source": "apify"})
    # card 2: only 5 obs -> dropped by min_obs
    for i, d in enumerate(months[:5]):
        rows.append({"card_id": 2, "date": d, "ungraded_price": 20.0 + i, "source": "apify"})
    # card 3: deep history but median price $0.50 -> dropped by min_price
    for i, d in enumerate(months):
        rows.append({"card_id": 3, "date": d, "ungraded_price": 0.5, "source": "apify"})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_filters_and_gap_handling():
    panel = build_panel(make_prices(), make_cfg())
    assert set(panel["card_id"].unique()) == {1}

    one = panel.sort_values("period").reset_index(drop=True)
    assert len(one) == 14  # full grid from first to global last period

    # hole at grid indices 5..8: ffill_limit=2 fills 5,6; 7,8 stay NaN
    assert not one.loc[[5, 6, 7, 8], "is_observed"].any()
    assert one.loc[5, "y"] == one.loc[4, "y"]
    assert one.loc[6, "y"] == one.loc[4, "y"]
    assert np.isnan(one.loc[7, "y"]) and np.isnan(one.loc[8, "y"])

    # staleness counts periods since last real observation, even past the limit
    np.testing.assert_array_equal(
        one.loc[4:10, "staleness"].to_numpy(), [0, 1, 2, 3, 4, 0, 0]
    )
    # log_y defined exactly where y is
    assert one["log_y"].notna().equals(one["y"].notna())


def test_observed_values_match_source():
    panel = build_panel(make_prices(), make_cfg())
    one = panel.sort_values("period").reset_index(drop=True)
    assert one.loc[0, "y"] == 10.0
    assert one.loc[13, "y"] == 23.0
    assert one.loc[4, "y"] == 14.0


def test_max_cards_cap():
    cfg = make_cfg(min_obs=3, min_price=0.1, max_cards=1)
    panel = build_panel(make_prices(), cfg)
    # cards 1 (10 obs) and 3 (14 obs) pass filters; cap keeps the most observed
    assert set(panel["card_id"].unique()) == {3}
