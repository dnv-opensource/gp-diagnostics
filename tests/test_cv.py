"""Unit tests for cv.py, focusing on correctness of LOO and multifold cross-validation residuals."""

import functools
import operator
import random

import gpytorch
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from gp_diagnostics.cv import check_folds_indices, check_lower_triangular, check_numeric_array, loo, multifold
from gp_diagnostics.utils.stats import split_test_train_fold
from tests.utils import (
    ExactGPModel,
    gpytorch_kernel_Matern,
    gpytorch_likelihood_gaussian,
    gpytorch_mean_constant,
)


def test_check_folds_indices_correct() -> None:
    """Test that check_folds_indices does not raise for correct input."""
    # Correct (should not do anything)
    check_folds_indices([[1, 3], [5, 6, 7], [0, 2], [4]], 8)


def test_check_folds_indices_nmax() -> None:
    """Test that check_folds_indices raises AssertionError when n_max is wrong."""
    with pytest.raises(AssertionError):
        check_folds_indices([[1, 3], [5, 6, 7], [0, 2], [4]], 6)  # Wrong n_max


def test_check_folds_indices_list() -> None:
    """Test that check_folds_indices raises AssertionError if not a list of lists."""
    with pytest.raises(AssertionError):
        # Not a list of lists
        check_folds_indices([1, 3, 5, 6, 7, 0, 2, 4], 8)  # type: ignore[reportArgumentType]


def test_check_folds_indices_int() -> None:
    """Test that check_folds_indices raises AssertionError if not a list of lists of ints."""
    # Wrong type: str (instead of int)
    with pytest.raises(AssertionError):
        check_folds_indices([["22"]], 8)  # type: ignore[reportArgumentType]
    # Wrong type: float (instead of int)
    with pytest.raises(AssertionError):
        check_folds_indices([[1, 0.5]], 8)  # type: ignore[reportArgumentType]


def test_check_folds_indices_exh() -> None:
    """Test that check_folds_indices raises if folds do not form a valid partition of range(n_max)."""
    with pytest.raises(AssertionError):
        check_folds_indices([[1, 3], [5, 7], [0, 2], [4]], 8)  # Not exhaustive indices

    with pytest.raises(AssertionError):
        check_folds_indices([[1, 3], [5, 6, 7, 9], [0, 2], [4]], 8)  # Not exhaustive indices


def test_check_folds_indices_empty() -> None:
    """Test that check_folds_indices raises AssertionError if any fold is empty."""
    with pytest.raises(AssertionError):
        check_folds_indices([[1, 3], [5, 6, 7], [0, 2], [], [4]], 8)


def test_check_lower_triangular() -> None:
    """Test that check_lower_triangular raises AssertionError for invalid inputs."""
    # These should be ok
    arr = np.array([[0.2, 0, 0], [3, 2.2, 0], [1, 2, 4]])
    check_lower_triangular(arr)

    arr = np.array([[1, 0], [2, 2.2]])
    check_lower_triangular(arr)

    arr = np.array([[1]])
    check_lower_triangular(arr)

    # These should raise error
    with pytest.raises(AssertionError):
        check_lower_triangular(np.array([[1, 2, 2.3], [0, 2.2, 3], [0.1, 0, 4]]))

    with pytest.raises(AssertionError):
        check_lower_triangular(np.ones(shape=(13, 14)))

    with pytest.raises(AssertionError):
        check_lower_triangular(np.array([[1, 2, 2.3], [0, 0.001, "a"]]))

    with pytest.raises(AssertionError):
        check_lower_triangular("a")  # type: ignore[reportArgumentType]


def test_check_numeric_array() -> None:
    """Test that check_numeric_array raises AssertionError when needed."""
    # These should be ok
    check_numeric_array(np.ones(4), 1)
    check_numeric_array(np.ones(shape=(2, 4)), 2)
    check_numeric_array(np.ones(shape=(2, 4, 6)), 3)
    check_numeric_array(np.array(4), 0)

    # These should raise error
    with pytest.raises(AssertionError):
        check_numeric_array(np.array(4), 1)

    with pytest.raises(AssertionError):
        check_numeric_array(np.ones(shape=(2, 4)), 1)

    with pytest.raises(AssertionError):
        check_numeric_array("a", 1)  # type: ignore[reportArgumentType]

    with pytest.raises(AssertionError):
        check_numeric_array(np.array([1, "1"]), 1)


def test_loo_1d() -> None:
    """Test the 1D example from Ginsbourger and Schaerer (2021).

    [D. Ginsbourger and C. Schaerer (2021). Fast calculation of Gaussian Process
        multiple-fold crossvalidation residuals and their covariances.
        arXiv:2101.03108]
    """
    # Covariance matrix and observations
    Y_train = np.array(
        [
            -0.6182,
            -0.3888,
            -0.3287,
            -0.2629,
            0.3614,
            0.1442,
            -0.0374,
            -0.0546,
            -0.0056,
            0.0529,
        ]
    )

    K = np.array(
        [
            [
                9.1475710e-02,
                5.4994639e-02,
                1.8560780e-02,
                4.8545646e-03,
                1.1045272e-03,
                2.3026903e-04,
                4.5216686e-05,
                8.5007923e-06,
                1.5461808e-06,
                2.7402149e-07,
            ],
            [
                5.4994639e-02,
                9.1475710e-02,
                5.4994628e-02,
                1.8560771e-02,
                4.8545646e-03,
                1.1045267e-03,
                2.3026903e-04,
                4.5216686e-05,
                8.5007923e-06,
                1.5461794e-06,
            ],
            [
                1.8560775e-02,
                5.4994617e-02,
                9.1475710e-02,
                5.4994628e-02,
                1.8560771e-02,
                4.8545622e-03,
                1.1045267e-03,
                2.3026903e-04,
                4.5216686e-05,
                8.5007923e-06,
            ],
            [
                4.8545646e-03,
                1.8560771e-02,
                5.4994628e-02,
                9.1475710e-02,
                5.4994628e-02,
                1.8560771e-02,
                4.8545646e-03,
                1.1045275e-03,
                2.3026903e-04,
                4.5216719e-05,
            ],
            [
                1.1045275e-03,
                4.8545646e-03,
                1.8560771e-02,
                5.4994628e-02,
                9.1475710e-02,
                5.4994617e-02,
                1.8560771e-02,
                4.8545646e-03,
                1.1045275e-03,
                2.3026903e-04,
            ],
            [
                2.3026903e-04,
                1.1045267e-03,
                4.8545622e-03,
                1.8560771e-02,
                5.4994617e-02,
                9.1475710e-02,
                5.4994635e-02,
                1.8560780e-02,
                4.8545660e-03,
                1.1045275e-03,
            ],
            [
                4.5216686e-05,
                2.3026884e-04,
                1.1045267e-03,
                4.8545646e-03,
                1.8560771e-02,
                5.4994635e-02,
                9.1475710e-02,
                5.4994639e-02,
                1.8560780e-02,
                4.8545660e-03,
            ],
            [
                8.5007923e-06,
                4.5216686e-05,
                2.3026903e-04,
                1.1045275e-03,
                4.8545646e-03,
                1.8560780e-02,
                5.4994639e-02,
                9.1475710e-02,
                5.4994617e-02,
                1.8560775e-02,
            ],
            [
                1.5461794e-06,
                8.5007923e-06,
                4.5216686e-05,
                2.3026903e-04,
                1.1045272e-03,
                4.8545660e-03,
                1.8560780e-02,
                5.4994628e-02,
                9.1475710e-02,
                5.4994639e-02,
            ],
            [
                2.7402149e-07,
                1.5461808e-06,
                8.5007923e-06,
                4.5216686e-05,
                2.3026903e-04,
                1.1045275e-03,
                4.8545660e-03,
                1.8560780e-02,
                5.4994639e-02,
                9.1475710e-02,
            ],
        ]
    )

    # From paper
    LOO_residuals_transformed_true = np.array(
        [
            -2.04393396,
            -0.07086865,
            -0.81325009,
            -0.33726709,
            2.07426555,
            -0.82414653,
            -0.05894939,
            -0.10534249,
            0.10176395,
            0.18906068,
        ]
    )

    LOO_mean_true = np.array(
        [
            -0.38365906,
            0.02787939,
            0.02736787,
            -0.29997396,
            0.36816096,
            -0.11112669,
            0.0047464,
            -0.01693309,
            -0.00649325,
            0.04409083,
        ]
    )

    LOO_cov_true = np.array(
        [
            [
                5.43868914e-02,
                -2.63221730e-02,
                1.02399085e-02,
                -3.40788392e-03,
                1.09471090e-03,
                -3.50791292e-04,
                1.12430032e-04,
                -3.61632192e-05,
                1.20125414e-05,
                -4.93005518e-06,
            ],
            [
                -2.63221748e-02,
                3.40218581e-02,
                -2.04288363e-02,
                8.01187754e-03,
                -2.66015902e-03,
                8.54554935e-04,
                -2.73940881e-04,
                8.81146188e-05,
                -2.92695640e-05,
                1.20124914e-05,
            ],
            [
                1.02399066e-02,
                -2.04288289e-02,
                3.19701731e-02,
                -1.97094288e-02,
                7.72963883e-03,
                -2.56570964e-03,
                8.24513379e-04,
                -2.65259878e-04,
                8.81142623e-05,
                -3.61629245e-05,
            ],
            [
                -3.40788229e-03,
                8.01187381e-03,
                -1.97094269e-02,
                3.17552164e-02,
                -1.96319763e-02,
                7.69940997e-03,
                -2.55653122e-03,
                8.24513205e-04,
                -2.73939862e-04,
                1.12429247e-04,
            ],
            [
                1.09471008e-03,
                -2.66015716e-03,
                7.72963790e-03,
                -1.96319763e-02,
                3.17334048e-02,
                -1.96248218e-02,
                7.69941136e-03,
                -2.56570987e-03,
                8.54552840e-04,
                -3.50789691e-04,
            ],
            [
                -3.50790826e-04,
                8.54554237e-04,
                -2.56570918e-03,
                7.69940997e-03,
                -1.96248218e-02,
                3.17334011e-02,
                -1.96319763e-02,
                7.72963837e-03,
                -2.66015413e-03,
                1.09470787e-03,
            ],
            [
                1.12429749e-04,
                -2.73940503e-04,
                8.24513205e-04,
                -2.55653122e-03,
                7.69941136e-03,
                -1.96319763e-02,
                3.17552052e-02,
                -1.97094250e-02,
                8.01186915e-03,
                -3.40787880e-03,
            ],
            [
                -3.61630882e-05,
                8.81144733e-05,
                -2.65259820e-04,
                8.24513321e-04,
                -2.56571034e-03,
                7.72963930e-03,
                -1.97094250e-02,
                3.19701731e-02,
                -2.04288270e-02,
                1.02399047e-02,
            ],
            [
                1.20125114e-05,
                -2.92695440e-05,
                8.81143787e-05,
                -2.73940270e-04,
                8.54553713e-04,
                -2.66015623e-03,
                8.01187288e-03,
                -2.04288345e-02,
                3.40218581e-02,
                -2.63221730e-02,
            ],
            [
                -4.93004836e-06,
                1.20124987e-05,
                -3.61630118e-05,
                1.12429487e-04,
                -3.50790157e-04,
                1.09470845e-03,
                -3.40788020e-03,
                1.02399066e-02,
                -2.63221730e-02,
                5.43868914e-02,
            ],
        ]
    )

    LOO_mean, LOO_cov, LOO_residuals_transformed = loo(K, Y_train)

    assert LOO_mean is not None
    assert LOO_cov is not None
    assert LOO_residuals_transformed is not None

    np.testing.assert_allclose(LOO_mean, LOO_mean_true, atol=1e-3)
    np.testing.assert_allclose(LOO_cov, LOO_cov_true, rtol=1e-05, atol=1e-08)
    np.testing.assert_allclose(LOO_residuals_transformed, LOO_residuals_transformed_true, atol=1e-3)


def generate_cv_data(
    N_DIM: int = 3,
    N_TRAIN: int = 100,
    N_DUPLICATE_X: int = 0,
    NUM_FOLDS: int = 8,
    NOISE_VAR: float = 0.0,
    *,
    SCRAMBLE: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[list[int]],
    NDArray[np.float64],
    torch.Tensor,
    torch.Tensor,
]:
    """Generate cross-validation data using a Matern GP and optional duplication/noise.

    Args:
      N_DIM: Dimensionality of input X.
      N_TRAIN: Number of training points.
      N_DUPLICATE_X: Number of duplicate points to generate.
      NUM_FOLDS: Number of folds for CV partition.
      NOISE_VAR: Observational noise variance.
      SCRAMBLE: Whether to shuffle folds or not.

    Returns:
      (cv_residual_means, cv_residual_vars, folds_indices, K_np, X_train, Y_train)
    """
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Generate N_TRAIN training Xs, with N_DUPLICATE_X duplicates
    X_train: torch.Tensor = torch.rand(size=(N_TRAIN - N_DUPLICATE_X, N_DIM))

    if N_DUPLICATE_X > 0:
        # pick some random duplicates
        dup_indices = rng.choice(np.arange(N_TRAIN - N_DUPLICATE_X), N_DUPLICATE_X)
        X_train = torch.cat([X_train, X_train[dup_indices]])

    # Define kernel and sample training data

    # Kernel
    ker: gpytorch.kernels.Kernel = gpytorch_kernel_Matern(
        outputscale=1.0,
        lengthscale=torch.ones(N_DIM) * 0.5,
    )

    # Covariance matrix; shape=(N_TRAIN, N_TRAIN)
    K = ker(X_train)

    # Distribution
    normal_rv: gpytorch.distributions.Distribution = gpytorch.distributions.MultivariateNormal(
        mean=torch.zeros(K.shape[0]),  # vector; shape=(N_TRAIN)
        covariance_matrix=K,  # matrix; shape=(N_TRAIN, N_TRAIN)
    )

    # Noise
    noise = 0.0
    if NOISE_VAR != 0.0:
        _noise_rv = gpytorch.distributions.MultivariateNormal(
            mean=torch.zeros(N_TRAIN),
            covariance_matrix=torch.eye(N_TRAIN) * NOISE_VAR,
        )
        noise = _noise_rv.sample()

    # Sample training data
    Y_train: torch.Tensor = normal_rv.sample() + noise

    # Make folds
    if NUM_FOLDS == N_TRAIN:
        folds_indices = [[i] for i in range(N_TRAIN)]
    else:
        # Note: The following sampling approach will not work if NUM_FOLDS is
        #   very big (wrt N_TRAIN), but we will only use it for some examples
        #   where NUM_FOLDS << N_TRAIN

        # Partition the data into folds
        folds_end = rng.multinomial(N_TRAIN, np.ones(NUM_FOLDS) / NUM_FOLDS).cumsum()
        folds_end = np.insert(folds_end, 0, 0, axis=0)
        folds_end_int = folds_end.astype(int)

        folds_indices = [list(range(folds_end_int[i], folds_end_int[i + 1])) for i in range(NUM_FOLDS)]

    if SCRAMBLE:
        _rnd_idx = rng.permutation(N_TRAIN)
        folds_indices = [list(_rnd_idx[idx]) for idx in folds_indices]

    check_folds_indices(folds_indices, N_TRAIN)

    # Fit GP: Create and fit GP model to training data
    _mean = gpytorch_mean_constant(0.0, fixed=True)
    _likelihood = gpytorch_likelihood_gaussian(variance=max(1e-6, NOISE_VAR), fixed=False)
    model: ExactGPModel = ExactGPModel(
        X_train,
        Y_train,
        _mean,
        ker,
        _likelihood,
        "",
        "",
    )
    model.eval_mode()

    # Manual multi-fold
    # Collect residual means and variances of all folds
    # (residual = observed - predicted)
    _residual_means = []
    _residual_vars = []
    for i in range(NUM_FOLDS):
        # Split on i-th fold
        fold_X_test, fold_X_train = split_test_train_fold(folds_indices, X_train, i)
        fold_Y_test, fold_Y_train = split_test_train_fold(folds_indices, Y_train, i)
        # Set training data
        model.set_train_data(inputs=fold_X_train, targets=fold_Y_train, strict=False)
        # Predict on test data
        m, v = model.predict(fold_X_test, latent=False)
        _residual_means.append((fold_Y_test - m).numpy())
        _residual_vars.append(v.numpy())

    # Concatenate and sort so that the residuals correspond to observation 1,2,3 etc.
    cv_residual_means: NDArray[np.float64] = np.block(_residual_means).flatten()
    cv_residual_vars: NDArray[np.float64] = np.block(_residual_vars).flatten()

    # reorder to match the original order
    folds_concat = functools.reduce(operator.iadd, folds_indices, [])
    idx_sort = list(np.argsort(folds_concat))
    cv_residual_means = cv_residual_means[idx_sort]
    cv_residual_vars = cv_residual_vars[idx_sort]

    return cv_residual_means, cv_residual_vars, folds_indices, K.to_dense().detach().numpy(), X_train, Y_train


def multitest_loo(
    N_DIM: int,
    N_TRAIN: int,
    NOISE_VAR: float,
    N_DUPLICATE_X: int,
) -> None:
    """Check that LOO formula matches manual loop-based approach for residuals/variances.

    Note: This does NOT check covariance and transformed residuals.
    """
    cv_residual_means, cv_residual_vars, folds, K, X_train, Y_train = generate_cv_data(
        N_DIM=N_DIM,
        N_TRAIN=N_TRAIN,
        N_DUPLICATE_X=N_DUPLICATE_X,
        NUM_FOLDS=N_TRAIN,
        NOISE_VAR=NOISE_VAR,
        SCRAMBLE=False,
    )

    # Compute residuals from Cholesky factor including jitter
    LOO_mean, LOO_cov, _ = loo(
        K,
        Y_train.numpy(),
        noise_variance=max(1e-6, NOISE_VAR),
    )
    assert LOO_mean is not None
    assert LOO_cov is not None
    LOO_var = LOO_cov.diagonal()

    np.testing.assert_allclose(LOO_var, cv_residual_vars, atol=1e-3)
    np.testing.assert_allclose(LOO_mean, cv_residual_means, atol=1e-3)


def test_loo_noiseless() -> None:
    """Test LOO with no observational noise and no duplicates."""
    multitest_loo(N_DIM=3, N_TRAIN=50, NOISE_VAR=0, N_DUPLICATE_X=0)


def test_loo_noise() -> None:
    """Test LOO with noise, no duplicates."""
    multitest_loo(N_DIM=3, N_TRAIN=50, NOISE_VAR=0.3, N_DUPLICATE_X=0)


def test_loo_noise_dupl() -> None:
    """Test LOO with noise and many duplicates."""
    multitest_loo(N_DIM=3, N_TRAIN=50, NOISE_VAR=0.3, N_DUPLICATE_X=30)


def multitest_multifold(
    N_DIM: int,
    N_TRAIN: int,
    NUM_FOLDS: int,
    NOISE_VAR: float,
    N_DUPLICATE_X: int,
    *,
    SCRAMBLE: bool,
) -> None:
    """Check that multifold formula matches manual loop-based approach for residuals/variances.

    Note: This does NOT check covariance and transformed residuals.
    """
    cv_residual_means, cv_residual_vars, folds, K, X_train, Y_train = generate_cv_data(
        N_DIM=N_DIM,
        N_TRAIN=N_TRAIN,
        N_DUPLICATE_X=N_DUPLICATE_X,
        NUM_FOLDS=NUM_FOLDS,
        NOISE_VAR=NOISE_VAR,
        SCRAMBLE=SCRAMBLE,
    )

    CV_mean, CV_cov, _ = multifold(
        K,
        Y_train.numpy(),
        folds,
        noise_variance=max(1e-6, NOISE_VAR),
    )
    assert CV_mean is not None
    assert CV_cov is not None
    CV_var = CV_cov.diagonal()

    np.testing.assert_allclose(CV_var, cv_residual_vars, atol=1e-4)
    np.testing.assert_allclose(CV_mean, cv_residual_means, atol=1e-3)


def test_multifold_noiseless() -> None:
    """Test multifold formula with no observational noise and no duplicates."""
    multitest_multifold(
        N_DIM=3,
        N_TRAIN=100,
        NUM_FOLDS=8,
        NOISE_VAR=0,
        N_DUPLICATE_X=0,
        SCRAMBLE=True,
    )


def test_multifold_noise() -> None:
    """Test multifold formula with observational noise and no duplicates."""
    multitest_multifold(
        N_DIM=3,
        N_TRAIN=100,
        NUM_FOLDS=8,
        NOISE_VAR=0.3,
        N_DUPLICATE_X=0,
        SCRAMBLE=True,
    )


def test_multifold_noise_dupl() -> None:
    """Test multifold formula with noise and duplicates."""
    multitest_multifold(
        N_DIM=3,
        N_TRAIN=100,
        NUM_FOLDS=8,
        NOISE_VAR=0.3,
        N_DUPLICATE_X=30,
        SCRAMBLE=True,
    )


def test_loo_multifold() -> None:
    """Test that multifold CV with fold size=1 matches LOO approach exactly."""
    N = 100
    NOISE_VAR = 0.23
    cv_residual_means, cv_residual_vars, folds, K, X_train, Y_train = generate_cv_data(
        N_DIM=2,
        N_TRAIN=N,
        N_DUPLICATE_X=20,
        NUM_FOLDS=N,
        NOISE_VAR=NOISE_VAR,
        SCRAMBLE=False,
    )

    CV_mean, CV_cov, CV_residuals_transformed = multifold(
        K,
        Y_train.numpy(),
        folds,
        noise_variance=max(1e-6, NOISE_VAR),
    )
    assert CV_mean is not None
    assert CV_cov is not None
    assert CV_residuals_transformed is not None

    LOO_mean, LOO_cov, LOO_residuals_transformed = loo(
        K,
        Y_train.numpy(),
        noise_variance=max(1e-6, NOISE_VAR),
    )
    assert LOO_mean is not None
    assert LOO_cov is not None
    assert LOO_residuals_transformed is not None

    np.testing.assert_allclose(CV_mean, LOO_mean)
    np.testing.assert_allclose(CV_cov, LOO_cov)
    np.testing.assert_allclose(CV_residuals_transformed, LOO_residuals_transformed)
