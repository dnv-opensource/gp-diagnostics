"""Provides functions for evaluating GP models via log marginal likelihood, pseudo-likelihood, and MSE.

Also includes convenience methods for log probability calculations under normal distributions.
"""

from __future__ import annotations

__all__ = ["evaluate_GP", "evaluate_GP_cholesky", "log_prob_normal", "log_prob_standard_normal"]

import numpy as np
import numpy.typing as npt

from gp_diagnostics.cv import (
    check_folds_indices,
    check_lower_triangular,
    check_numeric_array,
    loo_cholesky,
    multifold_cholesky,
)
from gp_diagnostics.utils.linalg import triang_solve, try_chol


def evaluate_GP(
    K: npt.NDArray[np.float64],
    Y_train: npt.NDArray[np.float64],
    *,
    folds: list[list[int]] | None = None,
    noise_variance: float = 0,
    check_args: bool = True,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Computes various GP evaluation metrics for the given covariance matrix and data.

    Optionally uses multifold CV if folds are provided, otherwise does LOO.

    Args:
        K: Prior covariance matrix, shape (n_samples, n_samples).
        Y_train: Training observations, shape (n_samples,).
        folds: List of index subsets for multifold CV. If None, does LOO.
        noise_variance: Observational noise variance (>= 0).
        check_args: If True, checks input validity.

    Returns:
        A dict with:
         - "log_marginal_likelihood": float
         - "log_pseudo_likelihood": float (sum of log-prob in the standard normal space for CV residuals)
         - "MSE": float
         - "residuals_mean": array of shape (n_samples,)
         - "residuals_var": array of shape (n_samples,)
         - "residuals_transformed": array of shape (n_samples,)
    """
    if check_args:
        check_numeric_array(Y_train, 1, "Y_train")
        check_numeric_array(K, 2, "K")
        assert K.shape[0] == Y_train.shape[0] and K.shape[1] == Y_train.shape[0], (
            f"The size of K {K.shape} is not compatible with Y_train {Y_train.shape}"
        )
        assert noise_variance >= 0, "noise_variance must be non-negative"
        if folds is not None:
            check_folds_indices(folds, Y_train.shape[0])

    L = try_chol(K, noise_variance, "evaluate_GP")
    if L is None:
        return {}

    return evaluate_GP_cholesky(L, Y_train, folds=folds, check_args=False)


def evaluate_GP_cholesky(
    L: npt.NDArray[np.float64],
    Y_train: npt.NDArray[np.float64],
    *,
    folds: list[list[int]] | None = None,
    check_args: bool = True,
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Performs GP evaluation metrics given a Cholesky factor of (K + noise*I).

    Args:
        L: Lower-triangular factor, shape (n_samples, n_samples).
        Y_train: Observations, shape (n_samples,).
        folds: If not None, multifold CV indices; else uses LOO.
        check_args: Whether to verify argument shapes.

    Returns:
        A dictionary of metrics:
         - "log_marginal_likelihood"
         - "log_pseudo_likelihood"
         - "MSE"
         - "residuals_mean"
         - "residuals_var"
         - "residuals_transformed"
    """
    if check_args:
        check_lower_triangular(L, "L")
        check_numeric_array(Y_train, 1, "Y_train")
        if folds is not None:
            check_folds_indices(folds, Y_train.shape[0])

    res: dict[str, npt.NDArray[np.float64] | float] = {}
    if folds is not None:
        mean, cov, residuals_transformed = multifold_cholesky(L, Y_train, folds, check_args=False)
    else:
        mean, cov, residuals_transformed = loo_cholesky(L, Y_train, check_args=False)

    res["log_marginal_likelihood"] = log_prob_normal(L, Y_train)
    res["log_pseudo_likelihood"] = log_prob_standard_normal(residuals_transformed)
    res["MSE"] = float(np.linalg.norm(mean))

    res["residuals_mean"] = mean
    res["residuals_var"] = cov.diagonal()
    res["residuals_transformed"] = residuals_transformed

    return res


def log_prob_normal(L: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> float:
    """Computes log probability of data Y under Gaussian with covariance L*L^T.

    Args:
        L: Lower-triangular factor, shape (n_samples, n_samples).
        Y: Data vector, shape (n_samples,).

    Returns:
        Scalar log probability of Y.
    """
    a = triang_solve(L, Y)
    return -0.5 * np.linalg.norm(a) ** 2 - np.log(L.diagonal()).sum() - (Y.shape[0] / 2) * np.log(2 * np.pi)


def log_prob_standard_normal(Y: npt.NDArray[np.float64]) -> float:
    """Computes log probability of data Y under a standard normal distribution.

    Args:
        Y: Data vector, shape (n_samples,).

    Returns:
        The scalar log probability of Y under N(0, I).
    """
    return -0.5 * np.linalg.norm(Y) ** 2 - (Y.shape[0] / 2) * np.log(2 * np.pi)
