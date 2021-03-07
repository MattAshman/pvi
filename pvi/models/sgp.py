import logging
import torch

from torch import distributions, nn, optim
from .base import Model
from pvi.utils.psd_utils import psd_inverse, add_diagonal

logger = logging.getLogger(__name__)


class SparseGaussianProcessModel(Model, nn.Module):
    """
    Sparse Gaussian process model using closed-form optimisation of q(u).
    """

    def __init__(self, inducing_locations, output_sigma=1., **kwargs):
        super(Model, self).__init__(**kwargs)
        super(nn.Module, self).__init__()

        # Construct inducing points and kernel.
        if self.hyperparameters["kernel_class"] is not None:
            self.kernel = self.hyperparameters["kernel_class"](
                **self.hyperparameters["kernel_params"]
            )
        else:
            raise ValueError("Kernel class not specified.")

        # Add private inducing points to parameters.
        self.register_parameter(
            "inducing_locations",
            nn.Parameter(inducing_locations, requires_grad=True))
        
        # Keep fixed, for now.
        self.register_buffer("output_sigma", torch.tensor(output_sigma))

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.] * self.hyperparameters["num_inducing"]),
            "np2": torch.tensor(
                [-.5] * self.hyperparameters["num_inducing"]).diag_embed()
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None,
            "num_inducing": 50,
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "print_epochs": 10
        }

    def forward(self, x, q):
        """
        Returns the (approximate) predictive posterior distribution of a
        sparse Gaussian process.
        :param x: The input locations to make predictions at.
        :param q: The distribution q(u).
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        mu = q.mean
        su = q.covariance_matrix

        # Parameters of prior.
        kxz = self.kernel(x, self.inducing_locations).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(
            self.inducing_locations, self.inducing_locations).evaluate(),
                           1e-4)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), 1e-4)

        ikzz = psd_inverse(kzz)

        # Predictive posterior.
        ppmu = kxz.matmul(ikzz).matmul(mu)
        ppcov = kxx + kxz.matmul(
            ikzz).matmul(su - kzz).matmul(ikzz).matmul(kzx)

        return distributions.MultivariateNormal(ppmu, covariance_matrix=ppcov)
    
    def likelihood_forward(self, x, theta=None):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Normal(x, self.output_sigma)
    
    def conjugate_update(self, data, q, t_i):
        """
        :param data: The local data to refine the model with.
        :param q: The parameters of the current global posterior q(θ).
        :param t_i: The parameters of the local factor t(θ).
        :return: q_new, t_i_new, the new global posterior and the 
        new local
        contribution.
        """
        # Set up data etc.
        x = data["x"]
        y = data["y"]

        # Parameters of prior.
        kxz = self.kernel(x, self.inducing_locations).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(
            self.inducing_locations, self.inducing_locations).evaluate(),
                           1e-4)

        lzz = kzz.cholesky()
        ikzz = psd_inverse(chol=lzz)

        # Closed form solution for optimum q(u).
        # TODO: this is incorrect. Assume cavity prior rather than p(u).
        sigma = self.output_sigma
        prec = (ikzz + sigma.pow(-2)
                * ikzz.matmul(kzx.matmul(kxz)).matmul(ikzz))
        np2 = -0.5 * prec
        np1 = sigma.pow(-2) * ikzz.matmul(kzx).matmul(y).squeeze(-1)

        q_new = {
            "np1": np1,
            "np2": np2,
        }

        t_i_new = {
            "np1": np1 - q["np1"] + t_i["np1"],
            "np2": np2 - q["np2"] + t_i["np2"],
        }
        
        return q_new, t_i_new

    def elbo(self, data, q, p):
        """
        Returns the ELBO of the sparse Gaussian process model under q(u), with 
        prior p(u).
        :param data: The local data to refine the model with.
        :param q: The parameters of the current global posterior q(θ).
        :param p: The parameters of the prior p(θ) (could be cavity).
        """
        # Set up data etc.
        x = data["x"]
        y = data["y"]

        # Parameters of prior.
        kxz = self.kernel(x, self.inducing_locations).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(
            self.inducing_locations, self.inducing_locations).evaluate(),
                           1e-4)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), 1e-4)

        # Derivation follows that in Hensman et al. (2013).
        ikzz = psd_inverse(kzz)
        a = kxz.matmul(ikzz)
        b = kxx - kxz.matmul(ikzz).matmul(kzx)

        sigma = self.output_sigma
        dist = self.likelihood_forward(a.matmul(q.mean))
        ll1 = dist.log_prob(y)
        ll2 = -0.5 * sigma.pow(-2) * (
                a.matmul(q.covariance_matrix).matmul(a.transpose(-1, -2)) + b)
        ll = ll1.sum() + ll2.sum()

        # Compute the KL divergence between current approximate
        # posterior and prior.
        kl = distributions.kl_divergence(q, p)
        
        elbo = ll - kl
        
        return elbo
