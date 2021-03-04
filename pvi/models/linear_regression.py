import torch

from torch import distributions
from .base import Model
from pvi.likelihoods.linear_regression import LinearRegressionLikelihood


class LinearRegressionModel(Model):
    def __init__(self, dim, output_sigma=1.):
        super().__init__(LinearRegressionLikelihood(output_sigma))

        self.dim = dim

    def get_default_parameters(self):
        return {
            "np1": torch.tensor([0.]*self.dim),
            "np2": torch.tensor([1.]*self.dim)
        }

    def get_default_hyperparameters(self):
        return {}

    def forward(self, x):
        """
        Returns the predictive posterior distribution of a Bayesian linear
        regression model.
        :param x: The input locations to make predictions at.
        :return: ∫ p(y | θ, x) p(θ | D) dθ.
        """
        w_prec = -2 * self.parameters["np2"]
        w_mu = torch.solve(self.parameters["np1"], w_prec)

        mu = x.T.matmul(w_mu)
        cov = x.T.matmul(torch.solve(x, w_prec))

        return distributions.MultivariateNormal(mu, cov)

    def fit(self, data, t_i):
        """
        :param data: The local data to refine the model with.
        :param t_i: The local contribution of the client.
        :return: t_i_new, the new local contribution.
        """
        np2_i_new = (-0.5 * self.likelihood.output_sigma ** (-2)
                     * data["x"].T.matmul(data["x"]))
        np1_i_new = (self.likelihood.output_sigma ** (-2)
                     * data["x"].T.matmul(data["y"]))

        self.parameters["np1"] = (self.parameters["np1"] - t_i["np1"]
                                  + np1_i_new)
        self.parameters["np2"] = (self.parameters["np2"] - t_i["np2"]
                                  + np2_i_new)

        t_i_new = {
            "np1": np1_i_new,
            "np2": np2_i_new
        }

        return t_i_new

    def sample(self, x, num_samples=1):
        """
        Samples the predictive posterior distirbutions of a Bayesian linear
        regression model.
        :param x: The input locations to make predictions at.
        :param num_samples: The number of samples to take.
        :return: A sample from the predictive posterior, ∫ p(y | θ,
        x) p(θ | D) dθ.
        """
        py_xD = self.forward(x)

        return py_xD.sample((num_samples,))