"""
Unit tests for linalg.py, verifying triangular solves, Cholesky-based inversion,
and trace product operations.
"""

import numpy as np

from gp_diagnostics.utils.linalg import chol_inv, mulinv_solve, mulinv_solve_rev, traceprod, triang_solve


def random_matrix(N: int, M: int, seed: int) -> np.ndarray:
    """
    Returns a random NxM matrix for testing, with a specified seed.
    """
    np.random.seed(seed)
    return np.random.uniform(size=(N, M))


def random_lower_triang_matrix(N: int, seed: int) -> np.ndarray:
    """
    Returns a random NxN lower triangular matrix by Cholesky of a random SPD matrix.
    """
    tmp = random_matrix(N, N, seed)
    return np.linalg.cholesky(tmp.dot(tmp.T))


def test_triang_solve() -> None:
    """
    Tests direct and transposed triangular solves with random data.
    """
    N = 10
    M = 4
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(N, M, 43)

    X = triang_solve(L, B, lower=True, trans=False)
    assert np.allclose(B, L.dot(X))

    X = triang_solve(L.T, B, lower=False, trans=False)
    assert np.allclose(B, L.T.dot(X))

    X = triang_solve(L, B, lower=True, trans=True)
    assert np.allclose(B, L.T.dot(X))

    X = triang_solve(L.T, B, lower=False, trans=True)
    assert np.allclose(B, L.dot(X))


def test_mulinv_solve() -> None:
    """
    Tests the function mulinv_solve for A=F*F^T with random data.
    """
    N = 10
    M = 4
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(N, M, 43)
    A = L.dot(L.T)

    X = mulinv_solve(L, B, lower=True)
    assert np.allclose(B, A.dot(X))

    X = mulinv_solve(L.T, B, lower=False)
    assert np.allclose(B, L.T.dot(L).dot(X))


def test_mulinv_solve_rev() -> None:
    """
    Tests the reversed (X*A=B) version of mulinv_solve.
    """
    N = 10
    M = 4
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(M, N, 43)
    A = L.dot(L.T)

    X = mulinv_solve_rev(L, B, lower=True)
    assert np.allclose(B, X.dot(A))

    X = mulinv_solve_rev(L.T, B, lower=False)
    assert np.allclose(B, X.dot(L.T.dot(L)))


def test_chol_inv() -> None:
    """
    Tests chol_inv for correctness vs. np.linalg.inv.
    """
    N = 10
    L = random_lower_triang_matrix(N, 42)
    A = L.dot(L.T)

    A_inv_true = np.linalg.inv(A)
    A_inv = chol_inv(L)
    assert np.allclose(A_inv_true, A_inv)


def test_traceprod() -> None:
    """
    Tests traceprod vs. direct diagonal sum for random data.
    """
    N = 10
    M = 8
    A = random_matrix(N, M, 42)
    B = random_matrix(M, N, 43)

    trace = traceprod(A, B)
    trace_true = A.dot(B).diagonal().sum()
    assert np.allclose(trace, trace_true)
