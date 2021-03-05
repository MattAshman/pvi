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
            "np1": torch.tensor([0.]*(self.hyperparameters["D"]+1)),
            "np2": torch.tensor(
                [1.]*(self.hyperparameters["D"]+1)).diag_embed(),
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None
        }

    def set_parameters(self, nat_params):
        """
        Sets the optimisable parameters. These are just the natural parameters,
        since we perform natural parameter updates in the optimisation.
        :param nat_params: Natural parameters of a multivariate Gaussian
        distribution.
        """
        self.register_parameter(
            "np1", nn.Parameter(nat_params["np1"], requires_grad=False))
        self.register_parameter(
            "np2", nn.Parameter(nat_params["np2"], requires_grad=False))

    def forward(self, x):
        """
        Returns the predictive posterior distribution of a Bayesian linear
        regression model.
        :param x: The input locations to make predictions at.
        :return: ∫ p(y | θ, x) p(θ | D) dθ.
        """
        prec = -2 * self.np2
        mu = torch.solve(self.np1.unsqueeze(-1), prec)[0].squeeze(-1)

        # Append 1 to end of x.
        x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        pp_mu = x_.matmul(mu)
        pp_var = x_.unsqueeze(-2).matmul(
            torch.solve(x_.unsqueeze(-1), prec)[0]).reshape(-1)

        return distributions.Normal(pp_mu, pp_var)

    def fit(self, data, t_i):
        """
        :param data: The local data to refine the model with.
        :param t_i: The local contribution of the client.
        :return: t_i_new, the new local contribution.
        """
        # Append 1 to end of x.
        x_ = torch.cat((data["x"], torch.ones(len(data["x"])).unsqueeze(-1)),
                       dim=1)
        np2_i_new = (-0.5 * self.likelihood.output_sigma ** (-2)
                     * x_.T.matmul(x_))
        np1_i_new = (self.likelihood.output_sigma ** (-2)
                     * x_.T.matmul(data["y"])).squeeze(-1)

        # Update model parameters.
        self.np1 = nn.Parameter(self.np1 - t_i["np1"] + np1_i_new,
                                requires_grad=False)
        self.np2 = nn.Parameter(self.np2 - t_i["np2"] + np2_i_new,
                                requires_grad=False)

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
        mu = torch.solve(nat_params["np1"].unsqueeze(-1), prec)[0].squeeze(-1)

        return distributions.MultivariateNormal(mu, precision_matrix=prec)

    @property
    def nat_params(self):
        """
        Returns the natural parameters, based on parameters included in self.
        :return: Natural parameters.
        """
        nat_params = {
            "np1": self.np1,
            "np2": self.np2,
        }
        return nat_params
