import logging
import torch

from torch import distributions, nn, optim
from .base import Model

logger = logging.getLogger(__name__)


class LogisticRegressionModel(Model, nn.Module):
    """
    Logistic regression model with a multivariate Gaussian approximate
    posterior.
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        super(nn.Module, self).__init__()

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.]*(self.hyperparameters["D"]+1)),
            "np2": torch.tensor([-.5]*(
                    self.hyperparameters["D"]+1)).diag_embed()
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None,
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "batch_size": 100,
            "num_elbo_samples": 1,
            "num_predictive_samples": 1,
            "print_epochs": 10,
        }

    def forward(self, x, q):
        """
        Returns the (approximate) predictive posterior distribution of a
        Bayesian logistic regression model.
        :param x: The input locations to make predictions at.
        :param q: The distribution q(θ).
        :return: ∫ p(y | θ, x) q(θ) dθ ≅ (1/M) Σ_m p(y | θ_m, x) θ_m ~ q(θ).
        """
        thetas = q.sample((self.hyperparameters["num_predictive_samples"],))

        comp = self.likelihood_forward(x, thetas)
        mix = distributions.Categorical(torch.ones(len(thetas),))

        return distributions.MixtureSameFamily(mix, comp)

    def likelihood_forward(self, x, theta):
        """
        Returns the model's likelihood p(y | θ, x).
        :param x: Input of shape (*, D).
        :param theta: Parameters of shape (*, D + 1).
        :return: Bernoulli distribution.
        """
        assert len(x.shape) in [1, 2], "x must be (*, D)."
        assert len(x.shape) in [1, 2], "theta must be (*, D)."

        if len(theta.shape) == 1:
            logits = (x.unsqueeze(-2).matmul(
                theta[:-1].unsqueeze(-1)).reshape(-1) + theta[-1])
        else:
            if len(x.shape) == 1:
                x_ = x.unsqueeze(0).repeat(len(theta), 1)
                logits = (x_.unsqueeze(-2).matmul(
                    theta[:, :-1].unsqueeze(-1)).reshape(-1) + theta[:, -1])
            else:
                x_ = x.unsqueeze(0).repeat(len(theta), 1, 1)
                theta_ = theta.unsqueeze(1).repeat(1, len(x), 1)
                logits = (x_.unsqueeze(-2).matmul(
                    theta_[..., :-1].unsqueeze(-1)).reshape(
                    len(theta), len(x)) + theta_[..., -1])

        return distributions.Bernoulli(logits=logits)

    def conjugate_update(self, data, q, t_i):
        """
        :param data: The local data to refine the model with.
        :param q: The parameters of the current global posterior q(θ).
        :param t_i: The parameters of the local factor t(θ).
        :return: q_new, t_i_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError
