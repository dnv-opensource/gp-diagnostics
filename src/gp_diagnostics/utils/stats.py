"""Statistical utilities for generating normal QQ data, partitioning folds, etc."""

from __future__ import annotations

__all__ = ["snorm_qq", "split_test_train_fold"]

import itertools
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def snorm_qq(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate standard normal QQ plot data, with approximate 95% confidence bands.

    Args:
        x: 1D data array for which to compute QQ plot info.

    Returns:
        (q_sample, q_snorm, q_snorm_upper, q_snorm_lower):
         - q_sample: Sorted sample quantiles from x.
         - q_snorm: Theoretical standard normal quantiles.
         - q_snorm_upper: Upper confidence band of theoretical quantiles.
         - q_snorm_lower: Lower confidence band of theoretical quantiles.
    """
    n = len(x)
    q_sample = np.sort(x)

    p = (np.arange(n) + 0.5) / n
    q_snorm = norm.ppf(p)

    # Approx confidence band
    k = 0.895 / (np.sqrt(n) * (1 - 0.01 / np.sqrt(n) + 0.85 / n))
    q_snorm_upper = norm.ppf(p + k)
    q_snorm_lower = norm.ppf(p - k)

    return q_sample, q_snorm, q_snorm_upper, q_snorm_lower


def split_test_train_fold(
    folds: list[list[int]],
    X: NDArray[np.float64],
    i: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split X into test and train arrays based on the i-th fold indices.

    Args:
        folds: List of index subsets.
        X: Array or matrix of shape (n_samples, ...).
        i: Fold index to designate as test.

    Returns:
        (X_test, X_train) with shapes based on folds[i].
    """
    idx_test = folds[i]
    idx_train = list(itertools.chain(*[folds[j] for j in range(len(folds)) if j != i]))
    return X[idx_test], X[idx_train]
