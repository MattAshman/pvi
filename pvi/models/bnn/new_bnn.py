from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from pvi.models import Model
from pvi.distributions import DistributionDict


class BNN(Model, ABC):

    conjugate_family = None

    def __init__(self, network: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = network

    def forward(self, x, q: DistributionDict, num_samples=10, **kwargs):
        """Returns the predictive distribution p(y | x) by drawing samples from q.

        Args:
            x (torch.tensor): Inputs to make predictions at.
            q (DistributionDict): Posterior distribution of network parameters.
            num_samples (int, optional): Number of samples to draw from q. Defaults to 10.

        Returns:
            torch.distributions.Distribution: Predictive distribution p(y | x).
        """
        thetas = [q.sample() for _ in range(num_samples)]
        qy = self.likelihood_forward(x, thetas)

        return qy

    def likelihood_forward(self, x, thetas):
        out = []
        for theta in thetas:
            # Set parameters of the network to those included in theta.
            for name, module in self.network.named_modules():
                for param_name, param in module.named_parameters():
                    attr_name = name + "." + param_name
                    if attr_name in theta:
                        param.data.copy_(theta[attr_name])

            # Forward pass.
            out.append(self.network(x))

        out = torch.stack(out)
        return self.likelihood_dist_from_tensor(out)

    @abstractmethod
    def likelihood_dist_from_tensor(self, tensor):
        raise NotImplementedError

    def expected_log_likelihood(self, data, q: DistributionDict, num_samples=1):
        thetas = [q.rsample() for _ in range(num_samples)]
        return self.likelhiood_log_prob(data, thetas).mean(0)


class RegressionBNN(BNN):
    def likelihood_dist_from_tensor(self, tensor):
        loc = tensor[..., : tensor.shape[-1] // 2]
        scale = torch.exp(tensor[..., tensor.shape[-1] // 2 :])

        return torch.distributions.Normal(loc=loc, scale=scale)


class ClassificationBNN(BNN):
    def likelihood_dist_from_tensor(self, tensor):
        return torch.distributions.Categorical(logits=tensor)
