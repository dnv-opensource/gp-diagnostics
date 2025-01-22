import gpytorch
import torch


# Define GP model and helper functions for GP creation
class ExactGPModel(gpytorch.models.ExactGP):
    """
    Model for standard GP regression
    """

    def __init__(self, train_x, train_y, mean_module, covar_module, likelihood, path, name):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        # For saving and loading
        self.path = path
        self.name = name
        self.param_fname = self.path + self.name + ".pth"

        # Mean and covariance functions
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def eval_mode(self):
        """
        Set model in evaluation mode
        """
        self.eval()
        self.likelihood.eval()

    def train_mode(self):
        """
        Set in training mode
        """
        self.train()
        self.likelihood.train()

    def predict(self, x, latent=True, CG_tol=0.1, full_cov=False):
        """
        Return mean and covariance at x

        Input:
        x         -      tensor of size dim * N containing N inputs
        latent    -      latent = True ->  using latent GP
                         latent = False -> using observed GP (incl. likelihood)
        CG_tol    -      Conjugate Gradient tolerance for evaluation
        full_cov  -      full_cov = False -> Return only diagonal (variances)

        Output:
        mean and covariance
        """

        with torch.no_grad(), gpytorch.settings.eval_cg_tolerance(CG_tol):
            # Latent distribution
            dist = self.__call__(x)

            # Observational distribution
            if not latent:
                dist = self.likelihood(dist)

            # Extract mean and covariance
            mean = dist.mean.cpu()
            var = dist.covariance_matrix.cpu() if full_cov else dist.variance.cpu()

            return mean, var

    def print_parameters(self):
        """
        Print actual (not raw) parameters
        """
        print("{:50} {}".format("Constant mean", self.mean_module.constant.item()))
        print("{:50} {}".format("Likelihood noise variance", self.likelihood.noise_covar.noise.item()))
        print("{:50} {}".format("Kernel lengthscale", self.covar_module.base_kernel.lengthscale.detach().numpy()[0]))
        print("{:50} {}".format("Kernel outputscale (variace)", self.covar_module.outputscale.item()))

    def save(self):
        """
        Save GP model parameters to self.path
        """
        print("Saving model to: ", self.param_fname)
        torch.save(self.state_dict(), self.param_fname)

    def load(self):
        """
        Load GP model parameters from self.path
        """
        print("Loading model from: ", self.param_fname)
        self.load_state_dict(torch.load(self.param_fname))


def gpytorch_kernel_Matern(var, ls, nu=2.5):
    """
    Return a Matern kernel with specified kernel variance (var) and lengthscales (ls)
    """
    ker_mat = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=len(ls))
    ker_mat.lengthscale = ls
    ker = gpytorch.kernels.ScaleKernel(ker_mat)
    ker.outputscale = var

    return ker


def gpytorch_mean_constant(val, fixed=True):
    """
    Return a constant mean function

    fixed = True -> Do not update mean function during training
    """
    mean = gpytorch.means.ConstantMean()
    mean.initialize(constant=val)
    mean.constant.requires_grad = not fixed

    return mean


def gpytorch_likelihood_gaussian(variance, variance_lb=1e-6, fixed=True):
    """
    Return a Gaussian likelihood

    fixed = True -> Do not update during training
    variance_lb = lower bound
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(variance_lb))
    likelihood.initialize(noise=variance)
    likelihood.requires_grad = not fixed

    return likelihood
