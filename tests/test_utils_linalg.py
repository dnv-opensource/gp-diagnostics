"""Unit tests for linalg.py, verifying triangular solves, Cholesky-based inversion, and trace product operations."""

import numpy as np

from gp_diagnostics.utils.linalg import (
    chol_inv,
    mulinv_solve,
    mulinv_solve_rev,
    symmetrify,
    traceprod,
    triang_solve,
)


def random_array(N: int, seed: int) -> np.ndarray:
    """Returns a random N-dimensional array with a given seed."""
    rng = np.random.default_rng(seed)
    return rng.random(size=N)


def random_matrix(N: int, M: int, seed: int) -> np.ndarray:
    """Returns a random NxM matrix with a given seed."""
    rng = np.random.default_rng(seed)
    return rng.random(size=(N, M))


def random_lower_triang_matrix(N: int, seed: int) -> np.ndarray:
    """Returns a random NxN lower triangular matrix by Cholesky of a random SPD matrix."""
    tmp = random_matrix(N, N, seed)
    return np.linalg.cholesky(tmp.dot(tmp.T))


def test_symmetrify() -> None:
    """Tests symmetrify in-place for random lower triangular matrices."""
    N = 10

    A = random_lower_triang_matrix(N, 77)
    symmetrify(A, upper=False)

    np.testing.assert_allclose(A - A.T, np.zeros((N, N)))

    A = random_lower_triang_matrix(N, 77)
    symmetrify(A.T, upper=True)

    np.testing.assert_allclose(A - A.T, np.zeros((N, N)))


def test_triang_solve() -> None:
    """Tests direct and transposed triangular solves with random data."""
    N = 10
    M = 4

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(N, M, 43)

    # Solve L*X = B and check
    X = triang_solve(L, B, lower=True, trans=False)
    assert X.shape == (N, M)
    np.testing.assert_allclose(B, L.dot(X))

    # Solve L.T*X = B and check
    X = triang_solve(L.T, B, lower=False, trans=False)
    assert X.shape == (N, M)
    np.testing.assert_allclose(B, L.T.dot(X))

    # Solve L.T*X = B and check
    X = triang_solve(L, B, lower=True, trans=True)
    assert X.shape == (N, M)
    np.testing.assert_allclose(B, L.T.dot(X))

    # Solve L*X = B and check
    X = triang_solve(L.T, B, lower=False, trans=True)
    assert X.shape == (N, M)
    np.testing.assert_allclose(B, L.dot(X))


def test_triang_solve_with_B_being_1D_array() -> None:
    """Tests direct and transposed triangular solves with a 1D RHS array and random data."""
    N = 10

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_array(N, 43)

    # Solve L*X = B and check
    X = triang_solve(L, B, lower=True, trans=False)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, L.dot(X))

    # Solve L.T*X = B and check
    X = triang_solve(L.T, B, lower=False, trans=False)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, L.T.dot(X))

    # Solve L.T*X = B and check
    X = triang_solve(L, B, lower=True, trans=True)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, L.T.dot(X))

    # Solve L*X = B and check
    X = triang_solve(L.T, B, lower=False, trans=True)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, L.dot(X))


def test_mulinv_solve() -> None:
    """Tests mulinv_solve for A=F*F^T with random data."""
    N = 10
    M = 4

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(N, M, 43)
    A = L.dot(L.T)

    # Solve A*X = B and check
    X = mulinv_solve(L, B, lower=True)
    assert X.shape == (N, M)
    np.testing.assert_allclose(B, A.dot(X))

    # Solve A*X = B and check
    X = mulinv_solve(L.T, B, lower=False)
    assert X.shape == (N, M)
    np.testing.assert_allclose(B, L.T.dot(L).dot(X))


def test_mulinv_solve_with_B_being_1D_array() -> None:
    """Tests mulinv_solve for A=F*F^T with a 1D RHS array and random data."""
    N = 10

    # Generate matrix and array
    L = random_lower_triang_matrix(N, 42)
    B = random_array(N, 43)
    A = L.dot(L.T)

    # Solve A*X = B and check
    X = mulinv_solve(L, B, lower=True)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, A.dot(X))

    # Solve A*X = B and check
    X = mulinv_solve(L.T, B, lower=False)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, L.T.dot(L).dot(X))


def test_mulinv_solve_rev() -> None:
    """Tests the reversed (X*A=B) version of mulinv_solve with random data."""
    N = 10
    M = 4

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(M, N, 43)
    A = L.dot(L.T)

    # Solve X*A = B and check
    X = mulinv_solve_rev(L, B, lower=True)
    assert X.shape == (M, N)
    np.testing.assert_allclose(B, X.dot(A))

    # Solve X*A = B and check
    X = mulinv_solve_rev(L.T, B, lower=False)
    assert X.shape == (M, N)
    np.testing.assert_allclose(B, X.dot(L.T.dot(L)))


def test_mulinv_solve_rev_with_B_being_1D_array() -> None:
    """Tests the reversed (X*A=B) version of mulinv_solve for A=F*F^T with a 1D RHS array and random data."""
    N = 10

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_array(N, 43)
    A = L.dot(L.T)

    # Solve X*A = B and check
    X = mulinv_solve_rev(L, B, lower=True)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, X.dot(A))

    # Solve X*A = B and check
    X = mulinv_solve_rev(L.T, B, lower=False)
    assert X.shape == (N,)
    np.testing.assert_allclose(B, X.dot(L.T.dot(L)))


def test_chol_inv() -> None:
    """Tests chol_inv for correctness vs. np.linalg.inv using random data."""
    N = 10

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    A = L.dot(L.T)

    # Invert and check
    A_inv_true = np.linalg.inv(A)
    A_inv = chol_inv(L)

    np.testing.assert_allclose(A_inv_true, A_inv)


def test_traceprod() -> None:
    """Tests traceprod vs. direct diagonal sum for random data."""
    N = 10
    M = 8

    # Generate matrices
    A = random_matrix(N, M, 42)
    B = random_matrix(M, N, 43)

    # Compute and check
    trace = traceprod(A, B)
    trace_true = A.dot(B).diagonal().sum()
    np.testing.assert_allclose(trace, trace_true)
