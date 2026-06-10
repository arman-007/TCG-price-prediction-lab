"""Shared runtime helpers: logging, seeding, timing.

(BLAS pinning lives in pricelab.threadlimits — a numpy-free module that
scripts must import before anything numeric.)
"""

from __future__ import annotations

import logging
import random
import sys
import time
import zlib

import numpy as np

from ..threadlimits import limit_blas_threads  # re-export

__all__ = ["limit_blas_threads", "setup_logging", "set_seeds", "stable_card_seed", "Timer"]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_card_seed(base_seed: int, card_id: int) -> int:
    """Deterministic per-card RNG seed, independent of worker scheduling."""
    return (base_seed * 1_000_003 + zlib.crc32(str(card_id).encode())) % (2**31)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self.start
