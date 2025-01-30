"""Helper classes and functions for testing, including a simple ExactGPModel and GPyTorch utilities."""

import gpytorch
import gpytorch.constraints
import torch


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
        """Initialize the ExactGPModel with training data, modules, and optional naming.

        Note: Overwriting the declaration of self.likelihood is necessary to
            make explicit to code linters that likelihood is not optional in our
            implementation, i.e. likelihood cannot be None. (This is different
            from the ExactGP base class implementation where likelihood can also
            be None.)
        """
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
        """Forward pass: return the latent GP at input x."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def eval_mode(self) -> None:
        """Set model in evaluation mode."""
        self.eval()
        self.likelihood.eval()

    def train_mode(self) -> None:
        """Set model in training mode."""
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
        """Return mean and covariance at x, optionally using the likelihood.

        Args:
          x: Tensor of shape (N, D) containing inputs to predict at.
          latent: Whether to return the latent GP distribution (True) or observed (False).
          CG_tol: Conjugate Gradient tolerance for evaluation.
          full_cov: If True, return full covariance matrix; else return only diagonal (variance).

        Returns:
          A tuple (mean, cov) as (Tensor, Tensor). If `full_cov` is False, cov will be shape (N,); otherwise shape
          (N, N).
        """
        with torch.no_grad(), gpytorch.settings.eval_cg_tolerance(CG_tol):
            dist = self.__call__(x)
            if not latent:
                maybe_obs = self.likelihood(dist)
                if isinstance(maybe_obs, torch.distributions.MultivariateNormal):
                    dist = maybe_obs

            # Extract mean and covariance
            assert isinstance(dist, gpytorch.distributions.MultivariateNormal)

            mean = dist.mean.cpu()
            var = dist.covariance_matrix.cpu() if full_cov else dist.variance.cpu()
        return mean, var

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
    """Return a scaled Matern kernel with specified hyperparams."""
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


def gpytorch_mean_constant(val: float, *, fixed: bool = True) -> gpytorch.means.Mean:
    """Return a ConstantMean function with a specified value.

    Args:
      val: The constant mean value.
      fixed: If True, do not update the mean during training.

    Returns:
      A ConstantMean instance set to 'val'.
    """
    mean = gpytorch.means.ConstantMean()
    mean.initialize(constant=val)
    assert isinstance(mean.constant, torch.Tensor)
    mean.constant.requires_grad = not fixed
    return mean


def gpytorch_likelihood_gaussian(
    variance: float,
    variance_lb: float = 1e-6,
    *,
    fixed: bool = True,
) -> gpytorch.likelihoods.Likelihood:
    """Return a Gaussian likelihood with optional fixed noise.

    TODO: Base type of likelihood is gpytorch.Module, not torch.Tensor.
        Natively, hence, likelihood does not have an attribute
        'requires_grad'.
        What the following code effectively does is to dynamically
        add an attribute with name='requires_grad' to the likelihood instance
        and assign it a boolean value.
        @AGRE / @ELD: Is this really what you intended, and is it necessary?
        CLAROS, 2022-11-01

    Args:
      variance: Noise variance.
      variance_lb: Lower bound for the noise variance.
      fixed: If True, do not update variance during training.

    Returns:
      A GaussianLikelihood instance.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(variance_lb))
    likelihood.initialize(noise=variance)
    likelihood.noise.requires_grad_(not fixed)
    return likelihood
