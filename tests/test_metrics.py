"""Unit tests for metrics.py, ensuring log-prob calculations match scipy.stats and main evaluation methods."""

import numpy as np
from scipy import stats

from gp_diagnostics.metrics import log_prob_normal, log_prob_standard_normal


def test_log_prob_normal() -> None:
    """Test that log_prob_normal matches scipy.stats for random data."""
    # Generate some data
    rng = np.random.default_rng(12)
    N = 17

    C = rng.random(size=(N, N))
    C = C.dot(C.T)
    Y = rng.multivariate_normal(mean=np.zeros(N), cov=C)

    # Compute likelihood
    L = np.linalg.cholesky(C)
    loglik = log_prob_normal(L, Y)
    loglik_scipy = stats.multivariate_normal.logpdf(Y, mean=np.zeros(N), cov=C)

    np.testing.assert_allclose(loglik, loglik_scipy)


def test_log_prob_standard_normal() -> None:
    """Test that log_prob_standard_normal matches scipy.stats for standard normal data."""
    # Generate some data
    rng = np.random.default_rng(12)
    N = 16

    Y = rng.multivariate_normal(mean=np.zeros(N), cov=np.eye(N))

    # Compute likelihood
    loglik = log_prob_standard_normal(Y)
    loglik_scipy = stats.multivariate_normal.logpdf(Y, mean=np.zeros(N), cov=np.eye(N))

    np.testing.assert_allclose(loglik, loglik_scipy)
