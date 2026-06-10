"""Read-only access to the main project's Postgres database.

The lab NEVER writes to the application DB: every connection is opened with
``default_transaction_read_only=on`` so an accidental INSERT/UPDATE fails
loudly instead of corrupting production data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

LAB_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(LAB_ROOT / ".env")

READ_ONLY = "-c default_transaction_read_only=on"


def get_connection():
    """Open a read-only connection. Two configuration styles, checked in order:

      1. DATABASE_URL=postgresql://user:pass@host:5432/dbname
         (single URL — handy for pointing the lab at staging/production)
      2. individual DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD vars
         (mirrors the main project's .env so creds can be copied verbatim)
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url, options=READ_ONLY)
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        options=READ_ONLY,
    )


def read_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame (column names from the cursor)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=cols)
