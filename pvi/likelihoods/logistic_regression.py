from torch import distributions
from torch.nn import functional as F
from .base import Likelihood


class LogisticRegressionLikelihood(Likelihood):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, theta):
        """
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
