from .base import ExponentialFamilyDistribution
from pvi.utils.psd_utils import psd_inverse

import torch


# =============================================================================
# Mean field gaussian distribution
# =============================================================================


class MeanFieldGaussianDistribution(ExponentialFamilyDistribution):
    
    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False):
        
        super().__init__(std_params=std_params,
                         nat_params=nat_params,
                         is_trainable=is_trainable)

    @property
    def torch_dist_class(self):
        return torch.distributions.Normal

    @property
    def mean_params(self):

        loc = self.std_params["loc"]
        scale = self.std_params["scale"]

        mp = {
            "m1": loc,
            "m2": scale ** 2 + loc ** 2,
        }

        return mp
        
    def _std_from_unc(self, unc_params):
        
        loc = unc_params["loc"]
        log_scale = unc_params["log_scale"]
        
        std = {
            "loc" : loc,
            "scale" : torch.exp(log_scale)
        }
        
        return std
    
    def _unc_from_std(self, std_params):
        
        loc = std_params["loc"].detach()
        scale = std_params["scale"].detach()
        
        unc = {
            "loc" : torch.nn.Parameter(loc),
            "log_scale" : torch.nn.Parameter(torch.log(scale))
        }
        
        return unc

    @classmethod
    def _nat_from_std(cls, std_params):
        
        loc = std_params["loc"]
        scale = std_params["scale"]
        
        nat = {
            "np1" : loc * scale ** -2,
            "np2" : -0.5 * scale ** -2
        }
        
        return nat
    
    @classmethod
    def _std_from_nat(cls, nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        std = {
            "loc" : - 0.5 * np1 / np2,
            "scale" : (- 0.5 / np2) ** 0.5
        }
        
        return std

    
# =============================================================================
# Multivariate gaussian distribution
# =============================================================================


class MultivariateGaussianDistribution(ExponentialFamilyDistribution):
    
    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False):
        
        super().__init__(std_params=std_params,
                         nat_params=nat_params,
                         is_trainable=is_trainable)

    @property
    def torch_dist_class(self):
        return torch.distributions.MultivariateNormal

    @property
    def mean_params(self):
        loc = self.std_params["loc"]
        covariance_matrix = self.std_params["covariance_matrix"]

        mp = {
            "m1": loc,
            "m2": covariance_matrix + loc.outer(loc),
        }

        return mp
        
    def _std_from_unc(self, unc_params):
        
        loc = unc_params["loc"]
        scale_tril = unc_params["scale_tril"]
        
        std = {
            "loc" : loc,
            "covariance_matrix" : torch.mm(scale_tril, scale_tril.T)
        }
        
        return std

    def _unc_from_std(self, std_params):

        loc = std_params["loc"].detach()
        cov = std_params["covariance_matrix"].detach()
        
        scale_tril = torch.cholesky(cov)
        
        unc = {
            "loc" : torch.nn.Parameter(loc),
            "scale_tril" : torch.nn.Parameter(scale_tril)
        }
        
        return unc
    
    @classmethod
    def _nat_from_std(cls, std_params):
        
        loc = std_params["loc"]
        cov = std_params["covariance_matrix"]
        
        nat = {
            "np1" : torch.solve(loc[:, None], cov).solution[:, 0],
            "np2" : -0.5 * psd_inverse(cov)
        }
        
        return nat
    
    @classmethod
    def _std_from_nat(cls, nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        prec = -2. * np2
        
        std = {
            "loc" : torch.solve(np1[:, None], prec).solution[:, 0],
            "covariance_matrix" : torch.inverse(prec)
        }
        
        return std


# =============================================================================
# Dirichlet distribution
# =============================================================================


class DirichletDistribution(ExponentialFamilyDistribution):
    
    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False):
        
        super().__init__(std_params=std_params,
                         nat_params=nat_params,
                         is_trainable=is_trainable)

    @property
    def torch_dist_class(self):
        return torch.distributions.Dirichlet

    @property
    def mean_params(self):
        raise NotImplementedError
        
    def _std_from_unc(self, unc_params):
        
        log_conc = unc_params["up1"]
        
        std = {
            "concentration" : torch.exp(log_conc)
        }
        
        return std

    def _unc_from_std(self, std_params):
        
        conc = std_params["concentration"].detach()
        log_conc = torch.exp(conc)
        
        unc = {
            "up1" : torch.nn.Parameter(log_conc)
        }
        
        return unc
    
    @classmethod
    def _nat_from_std(cls, std_params):
        
        conc = std_params["concentration"]
        
        nat = {
            "np1" : conc - 1.
        }
        
        return nat
    
    @classmethod
    def _std_from_nat(cls, nat_params):
        
        conc_minus_one = nat_params["np1"]
        
        std = {
            "concentration" : conc_minus_one + 1.
        }
        
        return std

    
# =============================================================================
# Multinomial distribution
# =============================================================================


class MultinomialDistribution(ExponentialFamilyDistribution):
    
    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False):
        
        super().__init__(std_params=std_params,
                         nat_params=nat_params,
                         is_trainable=is_trainable)

    @property
    def mean_params(self):
        raise NotImplementedError

    def _std_from_unc(self, unc_params):
        
        # First parameter is the number of trials and therefore not learnable
        up1 = unc_params["up1"]
        up2 = unc_params["up2"]
        
        p = torch.exp(up2)
        p = p / p.sum()
        
        std = {
            "total_count" : up1,
            "probs" : p,
        }
        
        return std
    
    def _unc_from_std(self, std_params):
        
        # First parameter is the number of trials and therefore not learnable
        sp1 = std_params["total_count"]
        sp2 = std_params["probs"]
        
        N = sp1
        log_p = torch.log(sp2)
        
        unc = {
            "up1" : torch.nn.Parameter(N, requires_grad=False).int(),
            "up2" : torch.nn.Parameter(log_p),
        }
        
        return unc

    @classmethod
    def _nat_from_std(cls, std_params):
        
        # First parameter is the number of trials and therefore not learnable
        sp1 = std_params["total_count"]
        sp2 = std_params["probs"]
        
        log_p = torch.log(sp2)
        
        nat = {
            "np1" : sp1,
            "np2" : log_p,
        }
        
        return nat

    @classmethod
    def _std_from_nat(cls, nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        p = torch.exp(np2)
        p = p / p.sum()
        
        std = {
            "total_count" : np1,
            "probs" : p
        }

        return std

    @property
    def torch_dist_class(self):
        return torch.distributions.Multinomial


# =============================================================================
# Gamma distribution
# =============================================================================


class GammaDistribution(ExponentialFamilyDistribution):

    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False):
        super().__init__(std_params=std_params,
                         nat_params=nat_params,
                         is_trainable=is_trainable)

    @property
    def mean_params(self):
        raise NotImplementedError

    def _std_from_unc(self, unc_params):
        log_alpha = unc_params["log_alpha"]
        log_beta = unc_params["log_beta"]

        concentration = log_alpha.exp()
        rate = 1 / log_beta.exp()

        std = {
            "concentration": concentration,
            "rate": rate
        }

        return std

    def _unc_from_std(self, std_params):
        concentration = std_params["concentration"].detach()
        rate = std_params["rate"].detach()

        unc = {
            "log_alpha": torch.nn.Parameter(concentration.log()),
            "log_beta": torch.nn.Parameter((1 / rate).log())
        }

        return unc

    @classmethod
    def _nat_from_std(cls, std_params):
        concentration = std_params["concentration"]
        rate = std_params["rate"]

        np1 = concentration - 1
        np2 = - 1 / rate

        nat = {
            "np1": np1,
            "np2": np2,
        }

        return nat

    @classmethod
    def _std_from_nat(cls, nat_params):
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]

        concentration = np1 + 1
        rate = -1 / np2

        std = {
            "concentration": concentration,
            "rate": rate
        }

        return std

    @property
    def torch_dist_class(self):
        return torch.distributions.Gamma
