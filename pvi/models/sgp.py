import torch

from torch import distributions, nn, optim
from gpytorch.kernels import ScaleKernel, RBFKernel
from .base import Model
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions.gp_distributions import \
    MultivariateGaussianDistributionWithZ

JITTER = 1e-6


class SparseGaussianProcessModel(Model, nn.Module):
    """
    Sparse Gaussian process model.
    """
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

        # Set eps.
        self.set_eps(self.eps)

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
            "kernel_class": lambda **kwargs: ScaleKernel(RBFKernel(**kwargs)),
            "kernel_params": {
                "outputscale": 1.,
                "lengthscale": 1.
            },
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "print_epochs": 10
        }

    def set_eps(self, eps):
        super().set_eps(eps)

        # Inverse softplus transformation.
        self.kernel.raw_outputscale = torch.log(
            torch.exp(self.eps["outputscale"]) - 1)
        self.kernel.base_kernel.raw_lengthscale = torch.log(
            torch.exp(self.eps["lengthscale"]) - 1)

    @staticmethod
    def get_default_eps():
        """
        :return: A default set of eps for the model.
        """
        return {
            "outputscale": 1.,
            "lengthscale": 1.,
        }

    def posterior(self, x, q):
        """
        Returns the  posterior distribution q(f) at locations x.
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


class SparseGaussianProcessRegression(SparseGaussianProcessModel):
    """
    Sparse Gaussian process regression model using closed-form optimisation of
    q(u).
    """

    conjugate_family = MultivariateGaussianDistributionWithZ

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_eps(self, eps):
        super().set_eps(eps)

        self.output_sigma = nn.Parameter(self.eps["output_sigma"])

    @staticmethod
    def get_default_eps():
        """
        :return: A default set of eps for the model.
        """
        return {
            **SparseGaussianProcessModel.get_default_eps(),
            "output_sigma": 1.,
        }

    def forward(self, x, q):
        return self.posterior(x, q)
    
    def likelihood_forward(self, x, theta=None):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Normal(x, self.output_sigma)
    
    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(u).
        :param t: The local factor t(u).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        if t is not None:
            assert torch.eq(q.inducing_locations, t.inducing_locations), \
                "q and t must share the same inducing locations for " \
                "conjugate update."

        # Set up data etc.
        x = data["x"]
        y = data["y"]
        z = q.inducing_locations

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
            t_new_np2 = -0.5 * sigma.pow(-2) * a.T.matmul(a)
            t_new_np1 = sigma.pow(-2) * a.T.matmul(y).squeeze(-1)
            t_new_nps = {"np1": t_new_np1, "np2": t_new_np2}

        if t is None:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] for k, v in q.nat_params.items()}

            q_new = type(q)(inducing_locations=z, nat_params=q_new_nps,
                            is_trainable=False)
            return q_new

        else:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] - t.nat_params[k]
                         for k, v in q.nat_params.items()}

            q_new = type(q)(inducing_locations=z, nat_params=q_new_nps,
                            is_trainable=False)
            t_new = type(t)(inducing_locations=z, nat_params=t_new_nps)
            return q_new, t_new

    def expected_log_likelihood(self, data, q):
        # Set up data etc.
        x = data["x"]
        y = data["y"]
        z = q.inducing_locations

        kzz = add_diagonal(self.model.kernel(z, z).evaluate(),
                           JITTER)
        ikzz = psd_inverse(kzz)
        kxz = self.model.kernel(x, z).evaluate()
        kxx = add_diagonal(self.model.kernel(x, x).evaluate(),
                           JITTER)

        a = kxz.matmul(ikzz)
        c = kxx - a.matmul(kxz.T)

        qf_loc = a.matmul(q.std_params["loc"])
        qf_cov = c + a.matmul(
            q.std_params["covariance_matrix"]).matmul(a.T)

        sigma = self.model.outputsigma
        dist = self.model.likelihood_forward(qf_loc)
        ll1 = dist.log_prob(y.squeeze())
        ll2 = -0.5 * sigma ** (-2) * qf_cov.diag()
        ll = ll1.sum() + ll2.sum()

        return ll

    def local_free_energy(self, data, q, t=None):
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


class SparseGaussianProcessClassification(SparseGaussianProcessModel):
    """
    Sparse Gaussian process classification model.
    """

    conjugate_family = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            **SparseGaussianProcessModel.get_default_hyperparameters(),
            "num_elbo_samples": 1,
            "num_predictive_samples": 1,
        }

    def forward(self, x, q):
        """
        Returns the (approximate) predictive posterior distribution of a
        sparse Gaussian process.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(u | z).
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        qf = self.posterior(x, q)
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

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError

    def expected_log_likelihood(self, data, q):
        raise NotImplementedError

    def local_free_energy(self, data, q, t=None):
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
