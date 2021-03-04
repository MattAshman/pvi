from torch import distributions
from torch.nn import functional as F
from .base import Likelihood


class LogisticRegressionLikelihood(Likelihood):
    def __init__(self):
        super().__init__()

    def forward(self, x, theta):
        return distributions.Bernoulli(
            logits=F.linear(x, theta[:-1], theta[-1]))
