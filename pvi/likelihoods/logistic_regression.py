from torch import distributions
from torch.nn import functional as F
from .base import Likelihood


class LogisticRegressionLikelihood(Likelihood):
    def __init__(self):
        super().__init__()

    def forward(self, x, theta):
        logits = (x.unsqueeze(-2).matmul(theta[:-1].unsqueeze(-1)).reshape(-1)
                  + theta[-1])
        return distributions.Bernoulli(logits=logits)
