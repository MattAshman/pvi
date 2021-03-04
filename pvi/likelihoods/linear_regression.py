import torch

from torch import distributions
from torch.nn import functional as F
from .base import Likelihood


class LinearRegressionLikelihood(Likelihood):
    def __init__(self, output_sigma=1.):
        super().__init__()

        # Keep fixed, for now.
        self.register_buffer("output_sigma", torch.tensor(output_sigma))

    def forward(self, x, theta):
        return distributions.Normal(
            F.linear(x, theta[:-1], theta[-1]), self.output_sigma)
