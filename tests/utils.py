"""
Helper classes and functions for testing, including a simple ExactGPModel
and convenience functions for flattening nested structures.
"""

import unittest
from collections.abc import Generator
from typing import Any

import gpytorch
import numpy as np
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    """
    A test GP model used for verifying correctness in unit tests.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.Likelihood,
        path: str,
        name: str,
    ) -> None:
        """
        Initializes the test GP model.

        Args:
            train_x: Training inputs, shape (n_data, n_dims).
            train_y: Training targets, shape (n_data,).
            mean_module: GPyTorch mean function.
            covar_module: GPyTorch kernel (covariance) function.
            likelihood: GPyTorch likelihood module.
            path: Filesystem path for saving/loading model parameters.
            name: Name of the model, used in file saving.
        """
        super().__init__(train_x, train_y, likelihood)

        self.path = path
        self.name = name
        self.param_fname = self.path + self.name + ".pth"

        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Defines the forward pass of the GP for the given input x.

        Args:
            x: Input tensor, shape (n_points, n_dims).

        Returns:
            MultivariateNormal distribution for the GP at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def eval_mode(self) -> None:
        """
        Sets model and likelihood to eval mode.
        """
        self.eval()
        self.likelihood.eval()

    def train_mode(self) -> None:
        """
        Sets model and likelihood to train mode.
        """
        self.train()
        self.likelihood.train()

    def predict(
        self,
        x: torch.Tensor,
        *,
        latent: bool = True,
        CG_tol: float = 0.1,
        full_cov: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and covariance of the predictions at x.

        Args:
            x: Inputs for prediction, shape (n_data, n_dims).
            latent: If True, use the latent GP, else use observed (likelihood).
            CG_tol: Tolerance for conjugate gradient evaluation.
            full_cov: If True, return full covariance matrix, else diagonal.

        Returns:
            (mean, covariance_or_variance) as CPU torch.Tensors.
        """
        with torch.no_grad(), gpytorch.settings.eval_cg_tolerance(CG_tol):
            dist = self.__call__(x)
            if not latent:
                dist = self.likelihood(dist)

            mean = dist.mean.cpu()
            var = dist.covariance_matrix.cpu() if full_cov else dist.variance.cpu()
            return mean, var


def gpytorch_kernel_Matern(
    var: float,
    ls: torch.Tensor,
    nu: float = 2.5,
) -> gpytorch.kernels.ScaleKernel:
    """
    Returns a Matern kernel with specified variance, lengthscales, and smoothness nu.

    Args:
        var: Output scale for the kernel.
        ls: Lengthscale(s) as a torch tensor.
        nu: Matern nu parameter (e.g. 2.5).

    Returns:
        ScaleKernel wrapping a MaternKernel.
    """
    ker_mat = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=len(ls))
    ker_mat.lengthscale = ls
    ker = gpytorch.kernels.ScaleKernel(ker_mat)
    ker.outputscale = var
    return ker


def gpytorch_mean_constant(val: float, *, fixed: bool = True) -> gpytorch.means.ConstantMean:
    """
    Creates a constant mean module with the specified initial value.

    Args:
        val: Constant mean value.
        fixed: If True, the constant is not trainable.

    Returns:
        ConstantMean module with the given initial constant.
    """
    mean = gpytorch.means.ConstantMean()
    mean.initialize(constant=val)
    mean.constant.requires_grad = not fixed
    return mean


def gpytorch_likelihood_gaussian(
    variance: float,
    variance_lb: float = 1e-6,
    *,
    fixed: bool = True,
) -> gpytorch.likelihoods.GaussianLikelihood:
    """
    Creates a Gaussian likelihood with the given noise variance.

    Args:
        variance: Initial noise variance.
        variance_lb: Lower bound on the noise parameter.
        fixed: If True, noise is not trainable.

    Returns:
        GaussianLikelihood with specified initial parameters.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(variance_lb))
    likelihood.initialize(noise=variance)
    likelihood.requires_grad = not fixed
    return likelihood


def flatten(data: Any) -> Generator[Any, None, None]:
    """
    Recursively flattens a nested iterable structure into a generator of scalar elements.

    Args:
        data: Possibly nested structure (lists, tuples, arrays).

    Yields:
        Each scalar element in the flattened structure.
    """
    if isinstance(data, str | bytes):
        yield data
        return
    try:
        iterator = iter(data)
    except TypeError:
        yield data
    else:
        for item in iterator:
            yield from flatten(item)


class TestFlattenFunction(unittest.TestCase):
    """
    Unit tests for the flatten() function ensuring correct flattening of nested structures.
    """

    def test_flatten_nested_list(self) -> None:
        nested = [[1, 2, [3, 4]], 5, (6, 7), np.array([8, 9])]
        expected = [1, 2, 3, 4, 5, 6, 7, np.int64(8), np.int64(9)]
        result = list(flatten(nested))
        assert result == expected

    def test_flatten_scalar(self) -> None:
        scalar = 10.5
        expected = [10.5]
        result = list(flatten(scalar))
        assert result == expected

    def test_flatten_empty_list(self) -> None:
        nested = []
        expected = []
        result = list(flatten(nested))
        assert result == expected

    def test_flatten_mixed_types(self) -> None:
        nested = [1, "two", [3, "four"], np.array([5.0, "six"])]
        expected = [1, "two", 3, "four", np.str_("5.0"), np.str_("six")]
        result = list(flatten(nested))
        assert result == expected

    def test_flatten_non_iterable_within_iterable(self) -> None:
        nested = [1, 2, None, [3, 4]]
        expected = [1, 2, None, 3, 4]
        result = list(flatten(nested))
        assert result == expected

    def test_flatten_strings_and_bytes(self) -> None:
        nested = ["hello", b"bytes", ["world", b"!"]]
        expected = ["hello", b"bytes", "world", b"!"]
        result = list(flatten(nested))
        assert result == expected

    def test_flatten_deeply_nested(self) -> None:
        nested = [[[[1]], 2], [[[3, [4]]]]]
        expected = [1, 2, 3, 4]
        result = list(flatten(nested))
        assert result == expected

    def test_flatten_custom_iterable(self) -> None:
        class CustomIterable:
            def __init__(self, items):
                self.items = items

            def __iter__(self):
                return iter(self.items)

        nested = [1, CustomIterable([2, 3]), [4, CustomIterable([5, [6]])]]
        expected = [1, 2, 3, 4, 5, 6]
        result = list(flatten(nested))
        assert result == expected
