import torch

from torch import distributions, nn, optim
from .base import Model
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions.exponential_family_factors import \
    MultivariateGaussianFactor
from pvi.distributions.exponential_family_distributions import \
    MultivariateGaussianDistribution

JITTER = 1e-6


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
        # TODO: these are shared across clients, hence requires_grad=False.
        self.register_parameter(
            "inducing_locations",
            nn.Parameter(inducing_locations, requires_grad=False))

        self.register_parameter("output_sigma", nn.Parameter(
            torch.tensor(output_sigma), requires_grad=True))

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
        su = q.distribution.covariance_matrix

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

        with torch.no_grad():
            # Parameters of prior.
            kxz = self.kernel(x, self.inducing_locations).evaluate()
            kzz = add_diagonal(self.kernel(
                self.inducing_locations, self.inducing_locations).evaluate(),
                               JITTER)

            lzz = kzz.cholesky()
            ikzz = psd_inverse(chol=lzz)

            sigma = self.output_sigma
            a = kxz.matmul(ikzz)

            # Closed form solution for optimum q(u).
            # New local parameters.
            np2_i_new = -0.5 * sigma.pow(-2) * a.T.matmul(a)
            np1_i_new = sigma.pow(-2) * a.T.matmul(y).squeeze(-1)

        # New model parameters.
        np1 = q.nat_params["np1"] - t_i.nat_params["np1"] + np1_i_new
        np2 = q.nat_params["np2"] - t_i.nat_params["np2"] + np2_i_new

        q_new_nps = {
            "np1": np1,
            "np2": np2,
        }
        q_new = MultivariateGaussianDistribution(nat_params=q_new_nps)

        t_i_new_nps = {
            "np1": np1_i_new,
            "np2": np2_i_new,
        }
        t_i_new = MultivariateGaussianFactor(t_i_new_nps)

        return q_new, t_i_new

    def elbo(self, data, q, p):
        """
        Returns the ELBO of the sparse Gaussian process model under q(u), with 
        prior p(u).
        :param data: The local data.
        :param q: The current global posterior q(θ).
        :param p: The prior p(θ) (could be cavity).
        :return: The evidence lower bound.
        """
        # Set up data etc.
        x = data["x"]
        y = data["y"]

        # Parameters of prior.
        kxz = self.kernel(x, self.inducing_locations).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(
            self.inducing_locations, self.inducing_locations).evaluate(),
                           JITTER)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), JITTER)

        # Derivation follows that in Hensman et al. (2013).
        ikzz = psd_inverse(kzz)
        a = kxz.matmul(ikzz)
        b = kxx - kxz.matmul(ikzz).matmul(kzx)
        qmu = q.distribution.mean
        qcov = q.distribution.covariance_matrix

        sigma = self.output_sigma
        dist = self.likelihood_forward(a.matmul(qmu))
        ll1 = dist.log_prob(y.squeeze())
        ll2 = -0.5 * sigma.pow(-2) * (a.matmul(qcov).matmul(a.T) + b)
        ll = ll1.sum() + ll2.sum()

        # Compute the KL divergence between current approximate
        # posterior and prior.
        kl = q.kl_divergence(p.distribution)

        elbo = ll - kl

        return elbo
