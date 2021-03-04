from torch import distributions
from .base import Likelihood


class HomoGaussian(Likelihood):
    def __init__(self, output_sigma):
        super().__init__()

        # Keep fixed, for now.
        self.register_buffer("output_sigma", output_sigma)

    def forward(self, x, theta):
        return distributions.Normal(theta, self.output_sigma)
