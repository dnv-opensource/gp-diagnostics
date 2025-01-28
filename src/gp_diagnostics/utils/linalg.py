"""Provides linear algebra utilities for Cholesky inversion, triangular solves, and related operations."""

from __future__ import annotations

__all__ = [
    "chol_inv",
    "mulinv_solve",
    "mulinv_solve_rev",
    "symmetrify",
    "traceprod",
    "triang_solve",
    "try_chol",
]

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray


def triang_solve(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    *,
    lower: bool = True,
    trans: bool = False,
) -> NDArray[np.float64]:
    """Solve a triangular system A*X = B or A^T*X = B, where A is triangular.

    Args:
        A: Triangular matrix of shape (N, N).
        B: Right-hand side, shape (N, NRHS).
        lower: If True, A is lower triangular; else upper.
        trans: If True, solve A^T * X = B; else A * X = B.

    Returns:
        A solution matrix X, shape (N, NRHS).
    """
    unitdiag = False
    lower_num = 1 if lower else 0
    trans_num = 1 if trans else 0
    unitdiag_num = 1 if unitdiag else 0

    A_fortran = np.asfortranarray(A)
    return linalg.lapack.dtrtrs(
        A_fortran,
        B,
        lower=lower_num,
        trans=trans_num,
        unitdiag=unitdiag_num,
    )[0]


def mulinv_solve(
    F: NDArray[np.float64],
    B: NDArray[np.float64],
    *,
    lower: bool = True,
) -> NDArray[np.float64]:
    """Solve A*X = B for A = F * F^T. Typically used when F is lower triangular.

    Args:
        F: Lower-triangular factor, shape (N, N).
        B: Right-hand side, shape (N, NRHS).
        lower: If True, F is lower triangular; else upper.

    Returns:
        The solution X, shape (N, NRHS).
    """
    tmp = triang_solve(F, B, lower=lower, trans=False)
    return triang_solve(F, tmp, lower=lower, trans=True)


def mulinv_solve_rev(
    F: NDArray[np.float64],
    B: NDArray[np.float64],
    *,
    lower: bool = True,
) -> NDArray[np.float64]:
    """Solve X*A = B for A = F*F^T. Typically used when F is lower triangular.

    Args:
        F: Lower-triangular factor, shape (N, N).
        B: Right-hand side, shape (NRHS, N).
        lower: If True, F is lower triangular; else upper.

    Returns:
        The solution X, shape (NRHS, N).
    """
    return mulinv_solve(F, B.T, lower=lower).T


def symmetrify(A: NDArray[np.float64], *, upper: bool = False) -> None:
    """Make matrix A symmetric by copying one triangle to the other in-place.

    Args:
        A: A square 2D numpy array.
        upper: If True, copy the upper triangle to the lower; else vice versa.
    """
    triu = np.triu_indices_from(A, k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]


def chol_inv(L: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return inverse of A = L * L^T, where L is lower triangular (via LAPACK dpotri).

    Args:
        L: Lower-triangular factor, shape (N, N).

    Returns:
        A_inv: The inverse of A, shape (N, N).
    """
    A_inv, _info = linalg.lapack.dpotri(L, lower=1)
    symmetrify(A_inv)
    return A_inv


def traceprod(A: NDArray[np.float64], B: NDArray[np.float64]) -> float:
    """Compute trace(A @ B) using Einstein summation.

    Args:
        A: 2D numpy array.
        B: 2D numpy array with shape (columns_A, something).

    Returns:
        Scalar trace of A*B.
    """
    return float(np.einsum("ij,ji->", A, B))


def try_chol(
    K: NDArray[np.float64],
    noise_variance: float,
    fun_name: str,
) -> NDArray[np.float64] | None:
    """Attempt Cholesky of (K + noise_variance*I). If fail, warn and return None.

    Args:
        K: Covariance matrix, shape (N, N).
        noise_variance: Noise variance to add to diagonal.
        fun_name: Label for logging/warnings.

    Returns:
        The lower-triangular Cholesky factor or None if decomposition fails.
    """
    A = K + np.eye(K.shape[0], dtype=np.float64) * noise_variance
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"Could not compute Cholesky in '{fun_name}'; matrix likely not PD. Returning None.",
            stacklevel=2,
        )
        try:
            min_eig = np.linalg.eig(A)[0].min()
            warnings.warn(f"Smallest eigenvalue: {min_eig}", stacklevel=2)
        except np.linalg.LinAlgError:
            warnings.warn("Could not compute smallest eigenvalue.", stacklevel=2)
        return None
