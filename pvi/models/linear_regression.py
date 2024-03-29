import torch

from torch import distributions, nn
from .base import Model
from pvi.distributions.exponential_family_distributions import \
    MultivariateGaussianDistribution


class LinearRegressionModel(Model, nn.Module):
    """
    Linear regression model with a Gaussian prior distribution.
    """
    
    conjugate_family = MultivariateGaussianDistribution

    def __init__(self, train_sigma=True, include_bias=True, **kwargs):
        self.include_bias = include_bias

        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.train_sigma = train_sigma
        if self.train_sigma:
            self.register_parameter("log_outputsigma", nn.Parameter(
                torch.tensor(self.hyperparameters["outputsigma"]).log(),
                requires_grad=True))
        else:
            self.register_buffer(
                "log_outputsigma",
                torch.tensor(self.hyperparameters["outputsigma"]).log())

        # Set ε after model is constructed.
        self.hyperparameters = self.hyperparameters

    def get_default_nat_params(self):
        if self.include_bias:
            return {
                "np1": torch.tensor([0.]*(self.config["D"] + 1)),
                "np2": torch.tensor(
                    [1.]*(self.config["D"] + 1)).diag_embed(),
            }
        else:
            return {
                "np1": torch.tensor([0.] * self.config["D"]),
                "np2": torch.tensor([1.] * self.config["D"]).diag_embed(),
            }

    def get_default_config(self):
        return {
            "D": None
        }

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._hyperparameters = {**self._hyperparameters, **hyperparameters}

        if hasattr(self, "log_outputsigma"):
            if self.train_sigma:
                self.log_outputsigma.data = \
                    self.hyperparameters["outputsigma"].log()
            else:
                self.log_outputsigma = \
                    self.hyperparameters["outputsigma"].log()

    def get_default_hyperparameters(self):
        """
        :return: A default set of ε for the model.
        """
        return {
            "outputsigma": torch.tensor(1.),
        }

    def forward(self, x, q, **kwargs):
        """
        Returns the predictive posterior distribution of a Bayesian linear
        regression model.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(θ).
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        prec = -2 * q.nat_params["np2"]
        mu = torch.solve(
            q.nat_params["np1"].unsqueeze(-1), prec)[0].squeeze(-1)

        if self.include_bias:
            x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        else:
            x_ = x

        ppmu = x_.matmul(mu)
        ppvar = x_.unsqueeze(-2).matmul(
            torch.solve(x_.unsqueeze(-1), prec)[0]).reshape(-1)

        return distributions.Normal(ppmu, ppvar**0.5)

    def likelihood_forward(self, x, theta, **kwargs):
        """
        Returns the model's likelihood p(y | θ, x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model, (*, D + 1).
        :return: p(y | θ, x)
        """
        assert len(x.shape) in [1, 2], "x must be (*, D)."
        assert len(x.shape) in [1, 2], "theta must be (*, D)."

        if self.include_bias:
            x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        else:
            x_ = x

        if len(theta.shape) == 1:
            mu = x.unsqueeze(-2).matmul(theta.unsqueeze(-1)).reshape(-1)
        else:
            if len(x.shape) == 1:
                x_r = x_.unsqueeze(0).repeat(len(theta), 1)
                mu = x_r.unsqueeze(-2).matmul(theta.unsqueeze(-1)).reshape(-1)
            else:
                x_r = x.unsqueeze(0).repeat(len(theta), 1, 1)
                theta_r = theta.unsqueeze(1).repeat(1, len(x_), 1)
                mu = x_r.unsqueeze(-2).matmul(theta_r.unsqueeze(-1)).reshape(
                    len(theta), len(x_))

        return distributions.Normal(mu, self.outputsigma)

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        x = data["x"]
        y = data["y"]

        if self.include_bias:
            x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        else:
            x_ = x

        with torch.no_grad():
            sigma = self.outputsigma

            # Closed-form solution.
            t_new_np2 = (-0.5 * sigma ** (-2) * x_.T.matmul(x_))
            t_new_np1 = (sigma ** (-2) * x_.T.matmul(y)).squeeze(-1)
            t_new_nps = {"np1": t_new_np1, "np2": t_new_np2}

        if t is None:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] for k, v in q.nat_params.items()}

            q_new = type(q)(nat_params=q_new_nps, is_trainable=False)
            return q_new, None

        else:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] - t.nat_params[k]
                         for k, v in q.nat_params.items()}

            q_new = type(q)(nat_params=q_new_nps, is_trainable=False)
            t_new = type(t)(nat_params=t_new_nps)
            return q_new, t_new

    def expected_log_likelihood(self, data, q, num_samples=1):
        n = data["x"].shape[0]
        sigma = self.outputsigma
        qmu = q.distribution.mean
        qcov = q.distribution.covariance_matrix

        x = data["x"]
        y = data["y"]

        if self.include_bias:
            x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        else:
            x_ = x

        mu = x_.matmul(qmu)
        cov = sigma ** 2 * torch.eye(n)
        dist = distributions.MultivariateNormal(mu, covariance_matrix=cov)

        dist_term = dist.log_prob(y).sum()
        trace_term = 0.5 * sigma ** (-2) * torch.trace(
            x_.matmul(qcov).matmul(x_.T))
        ell = dist_term - trace_term

        return ell

    def mll(self, data, q):
        """
        Returns the marginal log-likelihood of the linear regression model
        under the posterior q(θ).
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :return: p(y | x) = ∫ p(y | x, θ) q(θ) dθ = N(y; X * mw, X * Sw * X^T).
        """
        n = data["x"].shape[0]
        sigma = self.outputsigma
        qmu = q.distribution.mean
        qcov = q.distribution.covariance_matrix

        x = data["x"]
        y = data["y"]

        if self.include_bias:
            x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        else:
            x_ = x

        # Compute mean and covariance.
        ymu = x_.matmul(qmu)
        ycov = x_.matmul(qcov).matmul(x_.T) + sigma ** 2 * torch.eye(n)
        ydist = distributions.MultivariateNormal(ymu, covariance_matrix=ycov)

        return ydist.log_prob(y).sum()

    @property
    def outputsigma(self):
        return self.log_outputsigma.exp()
