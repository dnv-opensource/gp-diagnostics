"""Helper classes and functions for testing, including a simple ExactGPModel and GPyTorch utilities."""

import contextlib
from typing import TYPE_CHECKING, Any

import gpytorch
import gpytorch.constraints
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ExactGPModel(gpytorch.models.ExactGP):
    """Model for standard GP regression."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.Likelihood,
        path: str = "",
        name: str = "",
    ) -> None:
        # Note: Overwriting the declaration of self.likelihood is necessary to
        # make explicit to code linters that likelihood is not optional in our
        # implementation, i.e. likelihood cannot be None. (This is different
        # from the ExactGP base class implementation where likelihood can also
        # be None.)
        self.likelihood: gpytorch.likelihoods.Likelihood

        super().__init__(train_x, train_y, likelihood)
        assert self.likelihood is not None

        # For saving and loading
        self.path = path
        self.name = name
        self.param_fname = self.path + self.name + ".pth"

        # Mean and covariance functions
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def eval_mode(self) -> None:
        """Set model in evaluation mode."""
        self.eval()
        self.likelihood.eval()

    def train_mode(self) -> None:
        """Set in training mode."""
        self.train()
        self.likelihood.train()

    def predict(
        self,
        x: torch.Tensor,
        latent: bool = True,
        CG_tol: float = 0.1,
        full_cov: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and covariance at x.

        Input:
        x         -      tensor of size dim * N containing N inputs
        latent    -      latent = True ->  using latent GP
                         latent = False -> using observed GP (incl. likelihood)
        CG_tol    -      Conjugate Gradient tolerance for evaluation
        full_cov  -      full_cov = False -> Return only diagonal (variances)

        Output:
        mean and covariance
        """
        mean: torch.Tensor
        var: torch.Tensor
        with torch.no_grad(), gpytorch.settings.eval_cg_tolerance(CG_tol):
            # Latent distribution
            dist: torch.distributions.Distribution = self.__call__(x)

            # Observational distribution
            if not latent:
                _dist = self.likelihood(dist)
                if isinstance(_dist, torch.distributions.Distribution):
                    dist = _dist

            # Extract mean and covariance
            assert isinstance(dist, gpytorch.distributions.MultivariateNormal)
            # if isinstance(dist, gpytorch.distributions.MultivariateNormal):
            mean = dist.mean.cpu()
            var = (
                dist.covariance_matrix.cpu()  # type: ignore
                if full_cov
                else dist.variance.cpu()
            )

        return mean, var

    def print_parameters(self) -> None:
        """Print actual (not raw) parameters."""
        _constant_mean: torch.Tensor | str = "--"
        with contextlib.suppress(Exception):
            _constant_mean = self.mean_module.constant.item()  # type: ignore

        _noise: torch.Tensor | str = "--"
        with contextlib.suppress(Exception):
            _noise = self.likelihood.noise_covar.noise.item()  # type: ignore

        _lengthscale: NDArray[Any] | str = "--"
        with contextlib.suppress(Exception):
            _lengthscale = self.covar_module.base_kernel.lengthscale.detach().numpy()[0]

        _outputscale: torch.Tensor | str = "--"
        with contextlib.suppress(Exception):
            _outputscale = self.covar_module.outputscale.item()  # type: ignore


    def save(self) -> None:
        """Save GP model parameters to self.path."""
        torch.save(self.state_dict(), self.param_fname)

    def load(self) -> None:
        """Load GP model parameters from self.path."""
        self.load_state_dict(torch.load(self.param_fname))


def gpytorch_kernel_Matern(
    outputscale: float,
    lengthscale: torch.Tensor,
    nu: float = 2.5,
    lengthscale_constraint: gpytorch.constraints.Interval | None = None,
) -> gpytorch.kernels.Kernel:
    """Return a scaled Matern kernel with specified output scale and lengthscale."""
    lengthscale_constraint = lengthscale_constraint or gpytorch.constraints.Positive()
    ker_mat = gpytorch.kernels.MaternKernel(
        nu=nu,
        ard_num_dims=len(lengthscale),
        lengthscale_constraint=lengthscale_constraint,
    )
    ker_mat.lengthscale = lengthscale
    ker = gpytorch.kernels.ScaleKernel(ker_mat)
    ker.outputscale = outputscale

    return ker


def gpytorch_mean_constant(val: float, fixed: bool = True) -> gpytorch.means.Mean:
    """Return a constant mean function.

    fixed = True -> Do not update mean function during training
    """
    mean = gpytorch.means.ConstantMean()
    mean.initialize(constant=val)
    assert isinstance(mean.constant, torch.Tensor)
    mean.constant.requires_grad = not fixed

    return mean


def gpytorch_likelihood_gaussian(
    variance: float, variance_lb: float = 1e-6, fixed: bool = True
) -> gpytorch.likelihoods.Likelihood:
    """Return a Gaussian likelihood.

    fixed = True -> Do not update during training
    variance_lb = lower bound
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(variance_lb))
    likelihood.initialize(noise=variance)
    # @TODO: Base type of likelihood is gpytorch.Module, not torch.Tensor .
    #   Natively, hence, likelihood does not have an attribute
    #   'requires_grad'.
    #   What the following code effectively does is to dynamically
    #   add an attribute with name='requires_grad' to the likelihood instance
    #   and assign it a boolean value.
    #   @AGRE / @ELD: Is this really what you intended, and is it necessary?
    #   CLAROS, 2022-11-01
    likelihood.requires_grad = not fixed

    return likelihood
