"""
Unit tests for metrics.py, ensuring log-prob calculations match scipy.stats equivalents
and that the main evaluation methods function as expected.
"""

import numpy as np
from scipy import stats

from gp_diagnostics.metrics import log_prob_normal, log_prob_standard_normal


def test_log_prob_normal() -> None:
    """
    Tests that log_prob_normal matches scipy.stats.multivariate_normal.logpdf for random data.
    """
    np.random.seed(12)
    N = 17

    C = np.random.uniform(size=(N, N))
    C = C.dot(C.T)
    Y = np.random.multivariate_normal(mean=np.zeros(N), cov=C)

    L = np.linalg.cholesky(C)
    loglik = log_prob_normal(L, Y)
    loglik_scipy = stats.multivariate_normal.logpdf(Y, mean=np.zeros(N), cov=C)
    assert np.allclose(loglik, loglik_scipy)


def test_log_prob_standard_normal() -> None:
    """
    Tests that log_prob_standard_normal matches scipy.stats for standard normal data.
    """
    np.random.seed(12)
    N = 16

    Y = np.random.multivariate_normal(mean=np.zeros(N), cov=np.eye(N))
    loglik = log_prob_standard_normal(Y)
    loglik_scipy = stats.multivariate_normal.logpdf(Y, mean=np.zeros(N), cov=np.eye(N))

    assert np.allclose(loglik, loglik_scipy)
