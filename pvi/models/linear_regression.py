import torch

from torch import distributions, nn
from .base import Model


class LinearRegressionModel(Model, nn.Module):
    """
    Linear regression model with a Gaussian prior distribution.
    """
    def __init__(self, output_sigma=1., **kwargs):
        super(Model, self).__init__(**kwargs)
        super(nn.Module, self).__init__()

        # Keep fixed, for now.
        self.register_buffer("output_sigma", torch.tensor(output_sigma))

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.]*(self.hyperparameters["D"]+1)),
            "np2": torch.tensor(
                [1.]*(self.hyperparameters["D"]+1)).diag_embed(),
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
        :param q: The natural parameters of q(θ).
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        prec = -2 * q["np2"]
        mu = torch.solve(q["np1"].unsqueeze(-1), prec)[0].squeeze(-1)

        # Append 1 to end of x.
        x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        ppmu = x_.matmul(mu)
        ppvar = x_.unsqueeze(-2).matmul(
            torch.solve(x_.unsqueeze(-1), prec)[0]).reshape(-1)

        return distributions.Normal(ppmu, ppvar)

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

    def conjugate_update(self, data, q, t_i):
        """
        :param data: The local data to refine the model with.
        :param q: The parameters of the current global posterior q(θ).
        :param t_i: The parameters of the local factor t(θ).
        :return: q_new, t_i_new, the new global posterior and the new local
        contribution.
        """
        # Append 1 to end of x.
        x_ = torch.cat((data["x"], torch.ones(len(data["x"])).unsqueeze(-1)),
                       dim=1)
        np2_i_new = (-0.5 * self.output_sigma ** (-2) * x_.T.matmul(x_))
        np1_i_new = (self.output_sigma ** (-2)
                     * x_.T.matmul(data["y"])).squeeze(-1)

        # New model parameters.
        np1 = q["np1"] - t_i["np1"] + np1_i_new
        np2 = q["np2"] - t_i["np2"] + np2_i_new

        q_new = {
            "np1": np1,
            "np2": np2,
        }

        t_i_new = {
            "np1": np1_i_new,
            "np2": np2_i_new
        }

        return q_new, t_i_new
