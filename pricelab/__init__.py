"""pricelab: TCG card price forecasting research lab.

Standalone from the main Django project — the only shared surface is the
Postgres database, accessed read-only and snapshotted to parquet.
"""

__version__ = "0.1.0"
