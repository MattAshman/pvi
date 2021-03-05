import torch

from torch import distributions, nn
from .base import Likelihood


class HomoGaussian(Likelihood):
    def __init__(self, output_sigma):
        super().__init__()

        # Keep fixed, for now.
        self.register_buffer(
            "output_sigma",
            nn.Parameter(torch.tensor(output_sigma), requires_grad=False))

    def forward(self, x):
        return distributions.Normal(x, self.output_sigma)
