from torch import distributions
from .base import Likelihood


class HomoGaussian(Likelihood):
    def __init__(self, output_sigma):
        super().__init__()

        # Keep fixed, for now.
        self.register_buffer("output_sigma", output_sigma)

    def forward(self, x, theta):
        return distributions.Normal(theta, self.output_sigma)

    def log_prob(self, data, theta):
        py_x = distributions.Normal(theta, self.output_sigma)

        return py_x.log_prob(data["y"])

    def sample(self, x, theta, num_samples=1):
        py_x = distributions.Normal(theta, self.output_sigma)

        return py_x.sample((num_samples,))
