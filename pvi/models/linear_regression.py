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

    def __init__(self, output_sigma=1., **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.register_parameter("output_sigma", nn.Parameter(
            torch.tensor(output_sigma), requires_grad=True))

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.]*(self.hyperparameters["D"] + 1)),
            "np2": torch.tensor(
                [1.]*(self.hyperparameters["D"] + 1)).diag_embed(),
        }

    def get_default_hyperparameters(self):
        return {
            "D": None
        }

    def set_eps(self, eps):
        super().set_eps(eps)

    def get_default_eps(self):
        """
        :return: A default set of ε for the model.
        """
        return {}

    def forward(self, x, q):
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

        # Append 1 to end of x.
        x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        ppmu = x_.matmul(mu)
        ppvar = x_.unsqueeze(-2).matmul(
            torch.solve(x_.unsqueeze(-1), prec)[0]).reshape(-1)

        return distributions.Normal(ppmu, ppvar**0.5)

    def likelihood_forward(self, x, theta):
        """
        Returns the model's likelihood p(y | θ, x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model, (*, D + 1).
        :return: p(y | θ, x)
        """
        assert len(x.shape) in [1, 2], "x must be (*, D)."
        assert len(x.shape) in [1, 2], "theta must be (*, D)."

        if len(theta.shape) == 1:
            mu = (x.unsqueeze(-2).matmul(
                theta[:-1].unsqueeze(-1)).reshape(-1) + theta[-1])
        else:
            if len(x.shape) == 1:
                x_ = x.unsqueeze(0).repeat(len(theta), 1)
                mu = (x_.unsqueeze(-2).matmul(
                    theta[:, :-1].unsqueeze(-1)).reshape(-1) + theta[:, -1])
            else:
                x_ = x.unsqueeze(0).repeat(len(theta), 1, 1)
                theta_ = theta.unsqueeze(1).repeat(1, len(x), 1)
                mu = (x_.unsqueeze(-2).matmul(
                    theta_[..., :-1].unsqueeze(-1)).reshape(len(theta), len(x))
                      + theta_[..., -1])

        return distributions.Normal(mu, self.output_sigma)

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        # Append 1 to end of x.
        x_ = torch.cat((data["x"], torch.ones(len(data["x"])).unsqueeze(-1)),
                       dim=1)

        with torch.no_grad():
            sigma = self.output_sigma

            # Closed-form solution.
            t_new_np2 = (-0.5 * sigma ** (-2) * x_.T.matmul(x_))
            t_new_np1 = (sigma ** (-2) * x_.T.matmul(data["y"])).squeeze(-1)
            t_new_nps = {"np1": t_new_np1, "np2": t_new_np2}

        if t is None:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] for k, v in q.nat_params.items()}

            q_new = type(q)(nat_params=q_new_nps, is_trainable=False)
            return q_new

        else:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] - t.nat_params[k]
                         for k, v in q.nat_params.items()}

            q_new = type(q)(nat_params=q_new_nps, is_trainable=False)
            t_new = type(t)(nat_params=t_new_nps)
            return q_new, t_new

    def expected_log_likelihood(self, data, q):
        raise NotImplementedError("Not implemented, but is possible.")

    def mll(self, data, q):
        """
        Returns the marginal log-likelihood of the linear regression model
        under the posterior q(θ).
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :return: p(y | x) = ∫ p(y | x, θ) q(θ) dθ = N(y; X * mw, X * Sw * X^T).
        """
        n = data["x"].shape[0]
        sigma = self.output_sigma
        qmu = q.distribution.mean
        qcov = q.distribution.covariance_matrix

        # Append 1 to end of x.
        x_ = torch.cat((data["x"], torch.ones(len(data["x"])).unsqueeze(-1)),
                       dim=1)

        # Compute mean and covariance.
        ymu = x_.matmul(qmu)
        ycov = x_.matmul(qcov).matmul(x_.T) + sigma ** 2 * torch.eye(n)
        ydist = distributions.MultivariateNormal(ymu, covariance_matrix=ycov)

        return ydist.log_prob(data["y"]).sum()


class LinearRegressionModelNoBias(Model, nn.Module):
    """
    Linear regression model--with no bias!---with a Gaussian prior
    distribution.
    """

    conjugate_family = MultivariateGaussianDistribution

    def __init__(self, output_sigma=1., **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.register_parameter("output_sigma", nn.Parameter(
            torch.tensor(output_sigma), requires_grad=True))

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.] * (self.hyperparameters["D"] + 1)),
            "np2": torch.tensor(
                [1.] * (self.hyperparameters["D"] + 1)).diag_embed(),
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None
        }

    def forward(self, x, q):
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

        # Append 1 to end of x.
        ppmu = x.matmul(mu)
        ppvar = x.unsqueeze(-2).matmul(
            torch.solve(x.unsqueeze(-1), prec)[0]).reshape(-1)

        return distributions.Normal(ppmu, ppvar ** 0.5)

    def likelihood_forward(self, x, theta):
        """
        Returns the model's likelihood p(y | θ, x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model, (*, D + 1).
        :return: p(y | θ, x)
        """
        assert len(x.shape) in [1, 2], "x must be (*, D)."
        assert len(x.shape) in [1, 2], "theta must be (*, D)."

        if len(theta.shape) == 1:
            mu = x.unsqueeze(-2).matmul(theta.unsqueeze(-1)).reshape(-1)
        else:
            if len(x.shape) == 1:
                x_ = x.unsqueeze(0).repeat(len(theta), 1)
                mu = (x_.unsqueeze(-2).matmul(theta.unsqueeze(-1)).reshape(-1))
            else:
                x_ = x.unsqueeze(0).repeat(len(theta), 1, 1)
                theta_ = theta.unsqueeze(1).repeat(1, len(x), 1)
                mu = (x_.unsqueeze(-2).matmul(
                    theta_.unsqueeze(-1)).reshape(len(theta), len(x)))

        return distributions.Normal(mu, self.output_sigma)

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        x = data["x"]

        with torch.no_grad():
            sigma = self.output_sigma

            # Closed-form solution.
            t_new_np2 = (-0.5 * sigma ** (-2) * x.T.matmul(x))
            t_new_np1 = (sigma ** (-2) * x.T.matmul(data["y"])).squeeze(-1)
            t_new_nps = {"np1": t_new_np1, "np2": t_new_np2}

        if t is None:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] for k, v in q.nat_params.items()}

            q_new = type(q)(nat_params=q_new_nps, is_trainable=False)
            return q_new

        else:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] - t.nat_params[k]
                         for k, v in q.nat_params.items()}

            q_new = type(q)(nat_params=q_new_nps, is_trainable=False)
            t_new = type(t)(nat_params=t_new_nps)
            return q_new, t_new

    def expected_log_likelihood(self, data, q):
        raise NotImplementedError("Not implemented, but is possible.")

    def mll(self, data, q):
        """
        Returns the marginal log-likelihood of the linear regression model
        under the posterior q(θ).
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :return: p(y | x) = ∫ p(y | x, θ) q(θ) dθ = N(y; X * mw, X * Sw * X^T).
        """
        n = data["x"].shape[0]
        sigma = self.output_sigma
        qmu = q.distribution.mean
        qcov = q.distribution.covariance_matrix

        x = data["x"]

        # Compute mean and covariance.
        ymu = x.matmul(qmu)
        ycov = x.matmul(qcov).matmul(x.T) + sigma ** 2 * torch.eye(n)
        ydist = distributions.MultivariateNormal(ymu, covariance_matrix=ycov)

        return ydist.log_prob(data["y"]).sum()
