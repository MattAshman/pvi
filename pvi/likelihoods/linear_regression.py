from torch import distributions
from torch.nn import functional as F
from .base import Likelihood


class LogisticRegressionLikelihood(Likelihood):
    def __init__(self, output_sigma):
        super().__init__()

        # Keep fixed, for now.
        self.register_buffer("output_sigma", output_sigma)

    def forward(self, x, theta):
        w = theta["w"]
        b = theta["b"]

        mu = F.linear(x, w, b)
        return distributions.Normal(mu, self.output_sigma)

    def log_prob(self, data, theta):
        w = theta["w"]
        b = theta["b"]

        mu = F.linear(data["x"], w, b)
        py_x = distributions.Normal(mu, self.output_sigma)

        return py_x.log_prob(data["y"])

    def sample(self, x, theta, num_samples=1):
        w = theta["w"]
        b = theta["b"]

        mu = F.linear(x, w, b)
        py_x = distributions.Normal(mu, self.output_sigma)

        return py_x.sample((num_samples,))
