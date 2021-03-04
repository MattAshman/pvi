from torch import nn
from abc import ABC, abstractmethod


class Model(ABC, nn.Module):
    """
    An abstract class for probabilistic models defined by a likelihood
    p(y | θ, x) and (approximate) posterior q(θ).
    """
    def __init__(self, likelihood, nat_params=None, hyperparameters=None):
        super().__init__()

        self.likelihood = likelihood

        # Hyperparameters of the model.
        if hyperparameters is None:
            hyperparameters = {}

        # Parameters of the (approximate) posterior.
        if nat_params is None:
            nat_params = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.set_hyperparameters(hyperparameters)
        self.nat_params = self.get_default_nat_params()
        self.set_nat_params(nat_params)

    def set_nat_params(self, nat_params):
        self.nat_params = {**self.nat_params, **nat_params}

    def get_nat_params(self):
        return self.nat_params

    @abstractmethod
    def get_default_nat_params(self):
        """
        :return: A default set of natural parameters for the (approximate)
        posterior.
        """
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    def get_hyperparameters(self):
        return self.hyperparameters

    @abstractmethod
    def get_default_hyperparameters(self):
        """
        :return: A default set of hyperparameters for the model.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        """
        Returns the (approximate) predictive posterior.
        :param x: The input locations to make predictions at.
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, data, t_i):
        """
        :param data: The local data to refine the model with.
        :param t_i: The local contribution of the client.
        :return: t_i_new, the new local contribution.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, x, num_samples=1):
        """
        Samples the (approximate) predictive posterior.
        :param x: The input locations to make predictions at.
        :param num_samples: The number of samples to take.
        :return: A sample from the predictive posterior, ∫ p(y | θ, x) q(θ) dθ.
        """
        raise NotImplementedError

    @abstractmethod
    def get_distribution(self, parameters=None):
        """
        Returns the distribution defined by the parameters.
        :param parameters: Parameters of the distribution.
        :return: Distribution defined by the parameters.
        """
        raise NotImplementedError
