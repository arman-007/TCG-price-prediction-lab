"""DB -> local parquet snapshot.

All experiments run off an immutable local snapshot so results are
reproducible and the application DB sees exactly one read burst per refresh
instead of being hammered by every experiment iteration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .db import read_df

PRICE_COLUMNS = [
    "ungraded_price",
    "grade_7_price",
    "grade_8_price",
    "grade_9_price",
    "grade_9_5_price",
    "psa_10_price",
    "bgs_10_price",
    "cgc_10_price",
    "sgc_10_price",
]

PRICE_SQL = f"""
    SELECT card_id, date, source, {", ".join(PRICE_COLUMNS)}
    FROM cards_pricehistory
"""

CARD_SQL = """
    SELECT c.id AS card_id,
           c.pricecharting_id,
           c.product_name,
           c.console_name,
           c.genre,
           c.release_date,
           col.name AS collection_name,
           cat.name AS category_name
    FROM cards_card c
    LEFT JOIN cards_collection col ON col.id = c.collection_id
    LEFT JOIN cards_cardcategory cat ON cat.id = col.category_id
"""


def make_snapshot(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prices = read_df(PRICE_SQL)
    # psycopg2 returns Decimal; floats are what every model downstream wants.
    for col in PRICE_COLUMNS:
        prices[col] = pd.to_numeric(prices[col], errors="coerce").astype("float64")
    prices["date"] = pd.to_datetime(prices["date"])
    prices.to_parquet(out / "price_history.parquet", index=False)

    cards = read_df(CARD_SQL)
    cards["release_date"] = pd.to_datetime(cards["release_date"], errors="coerce")
    for col in ("product_name", "console_name", "genre", "collection_name", "category_name"):
        cards[col] = cards[col].astype("string")
    cards.to_parquet(out / "cards.parquet", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "price_rows": int(len(prices)),
        "cards": int(len(cards)),
        "min_date": str(prices["date"].min().date()),
        "max_date": str(prices["date"].max().date()),
        "sources": prices["source"].value_counts().to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def load_snapshot(snapshot_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    snap = Path(snapshot_dir)
    prices = pd.read_parquet(snap / "price_history.parquet")
    cards = pd.read_parquet(snap / "cards.parquet")
    return prices, cards
