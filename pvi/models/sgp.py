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
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

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
        :param q: The approximate posterior distribution q(u).
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        mu = q["distribution"].mean
        su = q["distribution"].covariance_matrix

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
        :param q: The current global posterior q(θ).
        :param t_i: The local factor t(θ).
        :return: q_new, t_i_new, the new global posterior and the new local
        contribution.
        """
        # Set up data etc.
        x = data["x"]
        y = data["y"]

        # Parameters of prior.
        kxz = self.kernel(x, self.inducing_locations).evaluate()
        kzz = add_diagonal(self.kernel(
            self.inducing_locations, self.inducing_locations).evaluate(),
                           1e-4)

        lzz = kzz.cholesky()
        ikzz = psd_inverse(chol=lzz)

        # Closed form solution for optimum q(u).
        sigma = self.output_sigma
        a = kxz.matmul(ikzz)

        # New local parameters.
        np2_i_new = -0.5 * sigma.pow(-2) * a.transpose(-1, -2).matmul(a)
        np1_i_new = sigma.pow(-2) * a.transpose(-1, -2).matmul(y).squeeze(-1)

        # New model parameters.
        np1 = q["nat_params"]["np1"] - t_i["nat_params"]["np1"] + np1_i_new
        np2 = q["nat_params"]["np2"] - t_i["nat_params"]["np2"] + np2_i_new

        q_new = {
            "nat_params": {
                "np1": np1,
                "np2": np2,
            }
        }

        t_i_new = {
            "nat_params": {
                "np1": np1_i_new,
                "np2": np2_i_new,
            }
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
        qmu = q["distribution"].mean
        qcov = q["distribution"].covariance_matrix

        sigma = self.output_sigma
        dist = self.likelihood_forward(a.matmul(qmu))
        ll1 = dist.log_prob(y)
        ll2 = -0.5 * sigma.pow(-2) * (
                a.matmul(qcov).matmul(a.transpose(-1, -2)) + b)
        ll = ll1.sum() + ll2.sum()

        # Compute the KL divergence between current approximate
        # posterior and prior.
        kl = distributions.kl_divergence(q["distribution"], p["distribution"])

        elbo = ll - kl

        # Alternative derivation.
        # F = log N(y; A * p.mean, sigma^2 I + A * p.cov * A^T)
        # - 0.5 * sigma^{-2} * Tr(Kxx - Kxz * Kzz^{-1} * Kzx)

        return elbo
