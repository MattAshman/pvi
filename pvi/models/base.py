from torch import nn
from abc import ABC, abstractmethod


class Model(ABC, nn.Module):
    """
    An abstract class for probabilistic models defined by a likelihood
    p(y | θ, x) and (approximate) posterior q(θ).
    """
    def __init__(self, likelihood, parameters=None, hyperparameters=None):
        super().__init__()

        self.likelihood = likelihood

        # Parameters of the (approximate) posterior.
        if parameters is None:
            parameters = {}

        # Hyperparameters of the model.
        if hyperparameters is None:
            hyperparameters = {}

        self.parameters = self.get_default_parameters()
        self.hyperparameters = self.get_default_hyperparameters()

        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

    def set_parameters(self, parameters):
        self.parameters = {**self.parameters, **parameters}

    def get_parameters(self):
        return self.parameters

    @abstractmethod
    def get_default_parameters(self):
        """
        :return: A default set of parameters for the (approximate) posterior.
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
