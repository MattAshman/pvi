import torch

from torch import distributions, nn, optim
from .base import Model
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions.gp_distributions import \
    MultivariateGaussianDistributionWithZ

JITTER = 1e-6


class SparseGaussianProcessRegression(Model, nn.Module):
    """
    Sparse Gaussian process regression model using closed-form optimisation of q(u).
    """

    conjugate_family = MultivariateGaussianDistributionWithZ

    def __init__(self, output_sigma=1., **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Construct inducing points and kernel.
        if self.hyperparameters["kernel_class"] is not None:
            self.kernel = self.hyperparameters["kernel_class"](
                **self.hyperparameters["kernel_params"]
            )
        else:
            raise ValueError("Kernel class not specified.")

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
        qu_loc = q.std_params["loc"]
        qu_cov = q.std_params["covariance_matrix"]
        z = q.inducing_locations

        # Parameters of prior.
        kxz = self.kernel(x, z).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(z, z).evaluate(), JITTER)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), JITTER)

        ikzz = psd_inverse(kzz)

        # Predictive posterior.
        qf_mu = kxz.matmul(ikzz).matmul(qu_loc)
        qf_cov = kxx + kxz.matmul(
            ikzz).matmul(qu_cov - kzz).matmul(ikzz).matmul(kzx)

        return distributions.MultivariateNormal(
            qf_mu, covariance_matrix=qf_cov)
    
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
        :param q: The current global posterior q(u).
        :param t_i: The local factor t(u).
        :return: q_new, t_i_new, the new global posterior and the new local
        contribution.
        """
        assert torch.eq(q.inducing_locations, t_i.inducing_locations), \
            "q and t_i must share the same inducing locations for conjugate " \
            "update."

        # Set up data etc.
        x = data["x"]
        y = data["y"]
        z = t_i.inducing_locations

        with torch.no_grad():
            # Parameters of prior.
            kxz = self.kernel(x, z).evaluate()
            kzz = add_diagonal(self.kernel(z, z).evaluate(), JITTER)

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
        q_new = type(q)(
            inducing_locations=q.inducing_locations,
            nat_params=q_new_nps,
        )

        t_i_new_nps = {
            "np1": np1_i_new,
            "np2": np2_i_new,
        }
        t_i_new = type(t_i)(
            inducing_locations=t_i.inducing_locations,
            nat_params=t_i_new_nps,
        )

        return q_new, t_i_new

    def local_free_energy(self, data, q, t):
        """
        Returns the local variational free energy (up to an additive constant)
        of the sparse Gaussian process model under q(u), with local factor
        t(u).
        :param data: The local data.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ) (could be cavity).
        :return: The evidence lower bound.
        """
        raise NotImplementedError


class SparseGaussianProcessClassification(Model, nn.Module):
    """
    Sparse Gaussian process classification model.
    """

    conjugate_family = None

    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Construct inducing points and kernel.
        if self.hyperparameters["kernel_class"] is not None:
            self.kernel = self.hyperparameters["kernel_class"](
                **self.hyperparameters["kernel_params"]
            )
        else:
            raise ValueError("Kernel class not specified.")

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
            "num_elbo_samples": 1,
            "num_predictive_samples": 1,
            "print_epochs": 10,
        }

    def forward(self, x, q):
        """
        Returns the (approximate) predictive posterior distribution of a
        sparse Gaussian process.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(u | z).
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        qu_loc = q.std_params["loc"]
        qu_cov = q.std_params["covariance_matrix"]
        z = q.inducing_locations

        # Parameters of prior.
        kxz = self.kernel(x, z).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(z, z).evaluate(), JITTER)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), JITTER)

        ikzz = psd_inverse(kzz)

        # Predictive posterior.
        qf_loc = kxz.matmul(ikzz).matmul(qu_loc)
        qf_cov = kxx + kxz.matmul(
            ikzz).matmul(qu_cov - kzz).matmul(ikzz).matmul(kzx)

        qf = distributions.MultivariateNormal(qf_loc, covariance_matrix=qf_cov)
        fs = qf.sample((self.hyperparameters["num_predictive_samples"],))

        comp = distributions.Bernoulli(logits=fs.T)
        mix = distributions.Categorical(torch.ones(len(fs),))

        return distributions.MixtureSameFamily(mix, comp)

    def likelihood_forward(self, x, theta):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Bernoulli(logits=theta)

    def conjugate_update(self, data, q, t_i):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t_i: The local factor t(θ).
        :return: q_new, t_i_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError

    def local_free_energy(self, data, q, t):
        """
        Returns the local variational free energy (up to an additive constant)
        of the sparse Gaussian process model under q(u), with local factor
        t(u).
        :param data: The local data.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ) (could be cavity).
        :return: The evidence lower bound.
        """
        raise NotImplementedError
