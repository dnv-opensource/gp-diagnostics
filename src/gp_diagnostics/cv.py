"""Implements leave-one-out (LOO) and multifold cross-validation for Gaussian Process models.

Includes optional fixed observational noise and uses Cholesky-based formulas for fast
calculation of residuals, following Ginsbourger and Schaerer (2021).
"""

from __future__ import annotations

__all__ = [
    "check_folds_indices",
    "check_lower_triangular",
    "check_numeric_array",
    "loo",
    "loo_cholesky",
    "multifold",
    "multifold_cholesky",
]

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from gp_diagnostics.utils import checks
from gp_diagnostics.utils.linalg import (
    chol_inv,
    mulinv_solve,
    triang_solve,
    try_chol,
)


def multifold(
    K: NDArray[np.float64],
    Y_train: NDArray[np.float64],
    folds: list[list[int]],
    *,
    noise_variance: float = 0,
    check_args: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | tuple[None, None, None]:
    """Compute multifold CV residuals for GP regression with optional noise variance.

    Args:
        K: GP prior covariance matrix, shape (n_samples, n_samples).
        Y_train: Training observations, shape (n_samples,).
        folds: List of list-index partitions. Each sublist is one fold's indices.
        noise_variance: Observational noise variance (0 = noiseless).
        check_args: If True, validates inputs (may raise assertions).

    Returns:
        (mean, cov, residuals_transformed), or (None, None, None) if Cholesky fails.
         - mean: Mean of multifold CV residuals, shape (n_samples,).
         - cov: Covariance matrix of multifold CV residuals, shape (n_samples, n_samples).
         - residuals_transformed: Residuals mapped into standard normal space, shape (n_samples,).

    Reference:
        D. Ginsbourger and C. Schaerer (2021). Fast calculation of Gaussian Process multiple-fold
        crossvalidation residuals and their covariances. arXiv:2101.03108
    """
    if check_args:
        check_numeric_array(Y_train, 1, "Y_train")
        check_numeric_array(K, 2, "K")
        assert K.shape[0] == Y_train.shape[0] and K.shape[1] == Y_train.shape[0], (
            f"The size of K {K.shape} is not compatible with Y_train {Y_train.shape}"
        )
        assert noise_variance >= 0, "noise_variance must be non-negative"
        check_folds_indices(folds, Y_train.shape[0])

    L = try_chol(K, noise_variance, "multifold")
    if L is None:
        return None, None, None

    return multifold_cholesky(L, Y_train, folds, check_args=False)


def multifold_cholesky(
    L: NDArray[np.float64],
    Y_train: NDArray[np.float64],
    folds: list[list[int]],
    *,
    check_args: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute multifold CV residuals given the Cholesky factor of the covariance matrix.

    Args:
        L: Lower-triangular Cholesky factor, shape (n_samples, n_samples).
        Y_train: Training observations, shape (n_samples,).
        folds: List of index partitions. Each sublist is one fold's indices.
        check_args: Whether to validate inputs.

    Returns:
        (mean, cov, residuals_transformed):
         - mean: Mean of multifold CV residuals.
         - cov: Covariance of multifold CV residuals.
         - residuals_transformed: Residuals mapped into standard normal space.

    Reference:
        D. Ginsbourger and C. Schaerer (2021). Fast calculation of Gaussian Process multiple-fold
        crossvalidation residuals and their covariances. arXiv:2101.03108
    """
    N_folds = len(folds)
    N_training = Y_train.shape[0]

    if check_args:
        check_lower_triangular(L, "L")
        check_numeric_array(Y_train, 1, "Y_train")
        check_folds_indices(folds, N_training)

    D = np.zeros((N_training, N_training), dtype=np.float64)
    D_inv_mean = np.zeros(N_training, dtype=np.float64)
    mean = np.zeros(N_training, dtype=np.float64)

    K_inv = chol_inv(L)
    K_inv_Y = mulinv_solve(L, Y_train)

    for i in range(N_folds):
        idx = np.ix_(folds[i], folds[i])
        block_chol = np.linalg.cholesky(K_inv[idx])
        D[idx] = chol_inv(block_chol)
        mean[folds[i]] = mulinv_solve(block_chol, K_inv_Y[folds[i]])
        D_inv_mean[folds[i]] = K_inv[idx].dot(mean[folds[i]])

    alpha = triang_solve(L, D)
    cov = alpha.T.dot(alpha)
    residuals_transformed = L.T.dot(D_inv_mean)
    return mean, cov, residuals_transformed


def loo(
    K: NDArray[np.float64],
    Y_train: NDArray[np.float64],
    *,
    noise_variance: float = 0,
    check_args: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | tuple[None, None, None]:
    """Compute Leave-One-Out (LOO) CV residuals for GP regression with optional noise variance.

    Args:
        K: GP prior covariance matrix, shape (n_samples, n_samples).
        Y_train: Training observations, shape (n_samples,).
        noise_variance: Observational noise variance (0 = noiseless).
        check_args: If True, validates inputs (may raise assertions).

    Returns:
        (mean, cov, residuals_transformed) or (None, None, None) if Cholesky fails:
         - mean: Mean of LOO residuals.
         - cov: Covariance matrix of LOO residuals.
         - residuals_transformed: LOO residuals mapped into standard normal space.
    """
    if check_args:
        check_numeric_array(Y_train, 1, "Y_train")
        check_numeric_array(K, 2, "K")
        assert K.shape[0] == Y_train.shape[0] and K.shape[1] == Y_train.shape[0], (
            f"The size of K {K.shape} is not compatible with Y_train {Y_train.shape}"
        )
        assert noise_variance >= 0, "noise_variance must be non-negative"

    L = try_chol(K, noise_variance, "loo")
    if L is None:
        return None, None, None

    return loo_cholesky(L, Y_train, check_args=False)


def loo_cholesky(
    L: NDArray[np.float64],
    Y_train: NDArray[np.float64],
    *,
    check_args: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute LOO residuals given the Cholesky factor of the covariance matrix.

    Args:
        L: Lower-triangular Cholesky factor (n_samples x n_samples).
        Y_train: Training observations, shape (n_samples,).
        check_args: If True, validates arguments.

    Returns:
        (mean, cov, residuals_transformed):
         - mean: Mean of LOO residuals.
         - cov: Covariance of LOO residuals.
         - residuals_transformed: LOO residuals mapped to standard normal space.

    References:
        O. Dubrule. Cross validation of kriging in a unique neighborhood.
        Journal of the International Association for Mathematical Geology, 15 (6):687-699, 1983.
    """
    if check_args:
        check_lower_triangular(L, "L")
        check_numeric_array(Y_train, 1, "Y_train")

    K_inv = chol_inv(L)

    var = 1.0 / K_inv.diagonal()
    mean = mulinv_solve(L, Y_train) * var

    D = np.eye(var.shape[0], dtype=np.float64) * var
    alpha = triang_solve(L, D)
    cov = alpha.T.dot(alpha)

    factor = np.eye(var.shape[0], dtype=np.float64) * (1.0 / var)
    residuals_transformed = L.T.dot(factor).dot(mean)

    return mean, cov, residuals_transformed


def check_folds_indices(folds: list[list[int]], n_max: int) -> None:
    """Check that folds is a valid partition of indices [0..n_max-1].

    Args:
        folds: List of sublists of integer indices.
        n_max: Total number of samples to index.

    Raises:
        AssertionError: If folds do not form a valid partition of range(n_max).
    """
    assert isinstance(folds, list), "'folds' must be a list of lists of integers"
    assert all(isinstance(x, list) for x in folds), "'folds' must be a list of lists of integers"
    assert [] not in folds, "'folds' must not contain empty subsets"

    all_elements_set = set(itertools.chain(*folds))
    assert all(np.issubdtype(type(x), np.integer) for x in all_elements_set), (
        "'folds' must contain only integer indices"
    )
    assert all_elements_set == set(range(n_max)), "The indices in 'folds' must form a partition of range(n_max)"


def check_lower_triangular(arr: NDArray[np.float64], argname: str = "arr") -> None:
    """Check that arr is a lower-triangular 2D numeric numpy array.

    Args:
        arr: Array to check.
        argname: Name of the argument for error messages.

    Raises:
        AssertionError: If arr is not a valid lower-triangular numeric array.
    """
    assert checks.is_numeric_np_array(arr), f"{argname} must be a numpy array with numeric elements"
    assert checks.is_square(arr), f"{argname} must be square"
    assert checks.is_lower_triang(arr), f"{argname} must be lower triangular"


def check_numeric_array(arr: NDArray[np.float64], dim: int, argname: str = "arr") -> None:
    """Check that arr is a numeric numpy array of specified dimension.

    Args:
        arr: Array to check.
        dim: Required dimensionality (1 for vector, 2 for matrix, etc.).
        argname: Argument name for error messages.

    Raises:
        AssertionError: If arr is not a numeric array of the requested dimension.
    """
    assert checks.is_numeric_np_array(arr), f"{argname} must be a numeric numpy array"
    assert len(arr.shape) == dim, f"{argname} must be {dim}-dimensional"


def _multifold_inv(
    K: NDArray[np.float64],
    Y_train: NDArray[np.float64],
    folds: list[list[int]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reference implementation using matrix inverse for testing multifold CV.

    Args:
        K: Covariance matrix, shape (n_samples, n_samples).
        Y_train: Observations, shape (n_samples,).
        folds: List of fold indices.

    Returns:
        (mean, cov, residuals_transformed) for multifold CV, used for testing only.
    """
    K_inv = np.linalg.inv(K)
    L = np.linalg.cholesky(K)

    N_training = Y_train.shape[0]
    N_folds = len(folds)

    K_inv_Y = K_inv.dot(Y_train)

    D = np.zeros((N_training, N_training), dtype=np.float64)
    D_inv = np.zeros((N_training, N_training), dtype=np.float64)
    mean = np.zeros(N_training, dtype=np.float64)
    D_inv_mean = np.zeros(N_training, dtype=np.float64)

    for i in range(N_folds):
        idx = np.ix_(folds[i], folds[i])
        block_inv = np.linalg.inv(K_inv[idx])
        D[idx] = block_inv
        D_inv[idx] = K_inv[idx]

        mean[folds[i]] = block_inv.dot(K_inv_Y[folds[i]])
        D_inv_mean[folds[i]] = D_inv[idx].dot(mean[folds[i]])

    cov = np.linalg.multi_dot([D, K_inv, D])
    residuals_transformed = L.T.dot(D_inv_mean)
    return mean, cov, residuals_transformed
