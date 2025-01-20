import warnings

import numpy as np
from scipy import linalg


def triang_solve(A, B, lower=True, trans=False):
    """
    Wrapper for lapack dtrtrs function
    DTRTRS solves a triangular system of the form
        A * X = B  or  A**T * X = B,
    where A is a triangular matrix of order N, and B is an N-by-NRHS
    matrix.  A check is made to verify that A is nonsingular.
    :param A: Matrix A(triangular)
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :param trans: calculate A**T * X = B (true) or A * X = B (false)

    :returns: Solution to A * X = B or A**T * X = B
    """
    unitdiag = False

    lower_num = 1 if lower else 0
    trans_num = 1 if trans else 0
    unitdiag_num = 1 if unitdiag else 0

    A = np.asfortranarray(A)

    return linalg.lapack.dtrtrs(A, B, lower=lower_num, trans=trans_num, unitdiag=unitdiag_num)[0]


def mulinv_solve(F, B, lower=True):
    """
    Solve A*X = B where A = F*F^{T}

    lower = True -> when F is LOWER triangular. This gives faster calculation
    """

    tmp = triang_solve(F, B, lower=lower, trans=False)
    return triang_solve(F, tmp, lower=lower, trans=True)


def mulinv_solve_rev(F, B, lower=True):
    """
    Reversed version of mulinv_solve

    Solves X*A = B where A = F*F^{T}

    lower = True -> when F is LOWER triangular. This gives faster calculation

    """
    return mulinv_solve(F, B.T, lower).T


def symmetrify(A, upper=False):
    """Create symmetric matrix from triangular matrix"""
    triu = np.triu_indices_from(A, k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]


def chol_inv(L):
    """
    Return inverse of matrix A = L*L.T where L is lower triangular
    Uses LAPACK function dpotri
    """
    A_inv, info = linalg.lapack.dpotri(L, lower=1)
    symmetrify(A_inv)
    return A_inv


def traceprod(A, B):
    """
    Calculate trace(A*B) for two matrices A and B
    """
    return np.sum(np.core.umath_tests.inner1d(A, B.T))


def try_chol(K, noise_variance, fun_name):
    """
    Try to compute the Cholesky decomposition of (K + noise_variance*I),
    and raise a warning if it fails.
    """
    A = K + np.eye(K.shape[0]) * noise_variance
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"Could not compute Cholesky decomposition in '{fun_name}'. The matrix is likely not positive definite. "
            "Consider adding jitter or a fallback approach. Returning None."
        )

        try:
            min_eig = np.linalg.eig(A)[0].min()
            warnings.warn(f"Smallest eigenvalue: {min_eig}")
        except np.linalg.LinAlgError:
            warnings.warn("Could not compute smallest eigenvalue.")

    return None
