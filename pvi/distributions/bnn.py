from typing import Any, Dict, Union, Optional

import torch
import torch.nn as nn
from pvi.distributions import (
    MeanFieldGaussianDistribution,
    MeanFieldGaussianFactor,
    DistributionDict,
    FactorDict,
)

BAYESIAN_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


class BNNMeanFieldGaussianDistribution(DistributionDict):
    """Distribution over the parameters of a BNN."""

    def __init__(
        self,
        network: nn.Module,
        state_dict: Optional[Dict[str, torch.tensor]] = {},
        fixed_modules: Optional[list[str]] = [],
        init_loc: float = 0.,
        init_std: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        distributions = {}
        for name, module in network.named_modules():
            cls = module.__class__
            if cls in BAYESIAN_MODULES:
                for param_name, param in module.named_parameters():
                    attr_name = name + "." + param_name
                    if attr_name in state_dict:
                        param.data.copy_(state_dict[attr_name].detach())
                        
                    if name not in fixed_modules:
                        param_loc = torch.full_like(param.data.detach(), init_loc)
                        param_std = torch.full_like(param_loc, init_std)

                        param_dist = MeanFieldGaussianDistribution(
                            std_params={"loc": param_loc, "scale": param_std}, **kwargs
                        )
                        distributions[attr_name] = param_dist

        self.distributions = distributions


class BNNMeanFieldGaussianFactor(FactorDict):
    """Factors over the parameters of a BNN."""

    def __init__(
        self,
        network: nn.Module,
        **kwargs,
    ):
        super().__init__()

        factors = {}
        for name, module in network.named_modules():
            cls = module.__class__
            if cls in BAYESIAN_MODULES:
                for param_name, param in module.named_parameters():
                    param_np1 = torch.zeros_like(param.data)
                    param_np2 = torch.zeros_like(param.data)

                    attr_name = name + "." + param_name

                    param_dist = MeanFieldGaussianFactor(
                        nat_params={"np1": param_np1, "np2": param_np2}, **kwargs
                    )
                    factors[attr_name] = param_dist

        self.factors = factors
