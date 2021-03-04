import torch

from torch import distributions, nn
from .base import Model
from pvi.likelihoods.linear_regression import LinearRegressionLikelihood


class LinearRegressionModel(Model):
    """
    Linear regression model with a Gaussian prior distribution.
    """
    def __init__(self, output_sigma=1., **kwargs):
        super().__init__(LinearRegressionLikelihood(output_sigma), **kwargs)

    def get_default_nat_params(self):
        return {
            "np1": nn.Parameter(
                torch.tensor([0.]*(self.hyperparameters["D"]+1)),
                requires_grad=False),
            "np2": nn.Parameter(
                torch.tensor([1.]*(self.hyperparameters["D"]+1)).diag_embed(),
                requires_grad=False)
        }

    def get_default_hyperparameters(self):
        return {
            "D": None
        }

    def forward(self, x):
        """
        Returns the predictive posterior distribution of a Bayesian linear
        regression model.
        :param x: The input locations to make predictions at.
        :return: ∫ p(y | θ, x) p(θ | D) dθ.
        """
        prec = -2 * self.nat_params["np2"]
        mu = torch.solve(self.nat_params["np1"], prec)

        pp_mu = x.T.matmul(mu)
        pp_cov = x.T.matmul(torch.solve(x, prec))

        return distributions.MultivariateNormal(pp_mu, pp_cov)

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

        # Update model parameters.
        self.nat_params["np1"] = (self.nat_params["np1"] - t_i["np1"]
                                  + np1_i_new)
        self.nat_params["np2"] = (self.nat_params["np2"] - t_i["np2"]
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
        pp = self.forward(x)

        return pp.sample((num_samples,))

    def get_distribution(self, nat_params=None):
        """
        Return a multivariate Gaussian distribution defined by parameters.
        :param nat_params: Natural parameters of a multivariate Gaussian
        distribution.
        :return: Multivariate Gaussian distribution defined by the parameters.
        """
        if nat_params is None:
            nat_params = self.nat_params

        prec = -2 * nat_params["np2"]
        mu = torch.solve(nat_params["np1"], prec)

        return distributions.MultivariateNormal(mu, precision_matrix=prec)
