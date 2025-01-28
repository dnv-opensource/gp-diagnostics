"""Utilities for checking array properties such as 'numeric', 'square', and 'lower triangular'."""

from __future__ import annotations

__all__ = ["is_lower_triang", "is_numeric_np_array", "is_square"]

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def is_numeric_np_array(arr: object) -> bool:
    """Check if arr is a numpy array with numeric dtype.

    Args:
        arr: Object to verify.

    Returns:
        True if arr is a numpy array with numeric type, else False.
    """
    if not isinstance(arr, np.ndarray):
        return False
    return arr.dtype.kind in {"b", "u", "i", "f", "c"}


def is_square(arr: NDArray[np.float64]) -> bool:
    """Check if arr is a 2D square numpy array.

    Args:
        arr: A numpy array to check.

    Returns:
        True if arr is 2D and arr.shape[0] == arr.shape[1], else False.
    """
    if arr.ndim != 2:  # noqa: PLR2004
        return False
    return arr.shape[0] == arr.shape[1]


def is_lower_triang(arr: NDArray[np.float64]) -> bool:
    """Check if arr is lower triangular.

    Args:
        arr: A 2D square numpy array.

    Returns:
        True if arr is lower triangular, else False.
    """
    idx = np.triu_indices_from(arr, k=1)
    return np.all(arr[idx] == 0)
