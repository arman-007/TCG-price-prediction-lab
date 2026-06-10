"""BLAS/OpenMP thread pinning. Deliberately numpy-free: these env vars only
take effect if set BEFORE numpy/scipy first load their BLAS, so scripts
import this module first, call limit_blas_threads(), then import the rest.
"""

import os


def limit_blas_threads(n: int = 1) -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))
