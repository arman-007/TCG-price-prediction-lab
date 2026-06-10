#!/usr/bin/env python
"""Pull a fresh read-only snapshot of the DB into data/raw/."""

from pricelab.threadlimits import limit_blas_threads

limit_blas_threads(1)

import argparse
import json
import logging

from pricelab.data.snapshot import make_snapshot
from pricelab.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    setup_logging()
    manifest = make_snapshot(args.out)
    logging.getLogger(__name__).info("snapshot done:\n%s", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
