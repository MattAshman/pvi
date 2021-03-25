"""
Identical to the multivariate Gaussian distributions in exponential_family_*.py
files, except they maintain their own inducing locations at which they are
defined.
"""

from pvi.distributions.exponential_family_distributions import \
    MultivariateGaussianDistribution
from pvi.distributions.exponential_family_factors import \
    MultivariateGaussianFactor

import torch.nn as nn


class MultivariateGaussianDistributionWithZ(MultivariateGaussianDistribution):
    def __init__(self,
                 inducing_locations=None,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False,
                 train_inducing=True):
        super().__init__(std_params, nat_params, is_trainable)

        self.train_inducing = train_inducing

        if is_trainable:
            self._inducing_locations = nn.Parameter(
                inducing_locations, requires_grad=self.train_inducing)
        else:
            self._inducing_locations = inducing_locations

    @property
    def inducing_locations(self):
        return self._inducing_locations

    @inducing_locations.setter
    def inducing_locations(self, value):
        if self.is_trainable:
            self._inducing_locations = nn.Parameter(
                value, requires_grad=self.train_inducing)
        else:
            self._inducing_locations = value

    def non_trainable_copy(self):

        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone()
                          for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone()
                              for k, v in self.std_params.items()}
            else:
                std_params = None

            if self._nat_params is not None:
                nat_params = {k: v.detach().clone()
                              for k, v in self.nat_params.items()}
            else:
                nat_params = None

        if self._inducing_locations is not None:
            inducing_locations = self.inducing_locations.detach().clone()
        else:
            inducing_locations = None

        return type(self)(inducing_locations, std_params, nat_params,
                          is_trainable=False)

    def trainable_copy(self):

        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone()
                          for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone()
                              for k, v in self.std_params.items()}
            else:
                std_params = None

            if self._nat_params is not None:
                nat_params = {k: v.detach().clone()
                              for k, v in self.nat_params.items()}
            else:
                nat_params = None

        if self._inducing_locations is not None:
            inducing_locations = self.inducing_locations.detach().clone()
        else:
            inducing_locations = None

        return type(self)(inducing_locations, std_params, nat_params,
                          is_trainable=True)


class MultivariateGaussianFactorWithZ(MultivariateGaussianFactor):
    def __init__(self, inducing_locations=None, nat_params=None,
                 train_inducing=True):
        MultivariateGaussianFactor.__init__(self, nat_params)

        self.train_inducing = train_inducing
        self._inducing_locations = inducing_locations

    def forward(self, x):
        raise NotImplementedError

    @property
    def inducing_locations(self):
        return self._inducing_locations

    @inducing_locations.setter
    def inducing_locations(self, value):
        self._inducing_locations = value
