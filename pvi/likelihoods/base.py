import torch.nn as nn

from abc import ABC, abstractmethod


class Likelihood(ABC, nn.Module):
    """
    An abstract class for likelihood functions of the form p(y | θ, x).
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, theta):
        """
        :param x: The input locations to make predictions at.
        :param theta: The latent variables of the model.
        :return: p(y | θ, x).
        """
        pass

    @abstractmethod
    def log_prob(self, data, theta):
        """
        Compute the log probability of the data under the likelihood.
        :param data: The data to compute the log likelihood of.
        :param theta: The latent variables of the model.
        :return: The log likelihood of the data.
        """
        pass

    @abstractmethod
    def sample(self, x, theta, num_samples=1):
        """
        Sample from the likelihood, p(y | θ, x).
        :param x: The input location to make predictions at.
        :param theta: The latent variables of the model.
        :param num_samples: The number of samples to take.
        :return: A sample from the likelihood, p(y | θ, x).
        """
        pass
