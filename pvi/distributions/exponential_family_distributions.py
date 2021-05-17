from .base import ExponentialFamilyDistribution
from pvi.utils.psd_utils import psd_inverse, psd_logdet

import torch
import numpy as np


# =============================================================================
# Mean field gaussian distribution
# =============================================================================


class MeanFieldGaussianDistribution(ExponentialFamilyDistribution):

    @property
    def torch_dist_class(self):
        return torch.distributions.Normal

    def log_a(self, nat_params=None):
        if nat_params is None:
            nat_params = self.nat_params

        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        log_a = -0.5 * np.log(np.pi) * len(np1)
        log_a += (- np1 ** 2 / (4 * np2) - 0.5 * (-2 * np2).log()).sum()

        return log_a

    @staticmethod
    def _std_from_unc(unc_params):
        
        loc = unc_params["loc"]
        log_scale = unc_params["log_scale"]
        
        std = {
            "loc": loc,
            "scale": torch.exp(log_scale)
        }
        
        return std

    @staticmethod
    def _unc_from_std(std_params):
        
        loc = std_params["loc"].detach()
        scale = std_params["scale"].detach()
        
        unc = {
            "loc": torch.nn.Parameter(loc),
            "log_scale": torch.nn.Parameter(torch.log(scale))
        }
        
        return unc

    @staticmethod
    def _nat_from_std(std_params):
        
        loc = std_params["loc"]
        scale = std_params["scale"]
        
        nat = {
            "np1": loc * scale ** -2,
            "np2": -0.5 * scale ** -2
        }
        
        return nat
    
    @staticmethod
    def _std_from_nat(nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        std = {
            "loc": - 0.5 * np1 / np2,
            "scale": (- 0.5 / np2) ** 0.5
        }
        
        return std

    @staticmethod
    def _mean_from_std(std_params):
        loc = std_params["loc"]
        scale = std_params["scale"]

        mp = {
            "m1": loc,
            "m2": scale ** 2 + loc ** 2,
        }

        return mp

    
# =============================================================================
# Multivariate gaussian distribution
# =============================================================================


class MultivariateGaussianDistribution(ExponentialFamilyDistribution):

    @property
    def torch_dist_class(self):
        return torch.distributions.MultivariateNormal

    def log_a(self, nat_params=None):
        if nat_params is None:
            nat_params = self.nat_params

        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        cov = psd_inverse(-2 * np2)
        log_a = -0.5 * np.log(np.pi) * len(np1)
        log_a += (0.25 * np1.dot(cov.matmul(np1))
                  - 0.5 * (psd_logdet(-2 * np2)))

        return log_a

    @staticmethod
    def _std_from_unc(unc_params):
        
        loc = unc_params["loc"]
        scale_tril = unc_params["scale_tril"]
        
        std = {
            "loc": loc,
            "covariance_matrix": torch.mm(scale_tril, scale_tril.T)
        }
        
        return std

    @staticmethod
    def _unc_from_std(std_params):

        loc = std_params["loc"].detach()
        cov = std_params["covariance_matrix"].detach()
        
        scale_tril = torch.cholesky(cov)
        
        unc = {
            "loc": torch.nn.Parameter(loc),
            "scale_tril": torch.nn.Parameter(scale_tril)
        }
        
        return unc
    
    @staticmethod
    def _nat_from_std(std_params):
        
        loc = std_params["loc"]
        cov = std_params["covariance_matrix"]
        prec = psd_inverse(cov)
        
        nat = {
            "np1": prec.matmul(loc),
            "np2": -0.5 * prec
        }
        
        return nat
    
    @staticmethod
    def _std_from_nat(nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        prec = -2. * np2
        cov = psd_inverse(prec)
        loc = cov.matmul(np1)
        
        std = {
            "loc": loc,
            "covariance_matrix": cov
        }
        
        return std

    @staticmethod
    def _mean_from_std(std_params):
        loc = std_params["loc"]
        cov = std_params["covariance_matrix"]

        mp = {
            "m1": loc,
            "m2": cov + loc.outer(loc),
        }

        return mp


# =============================================================================
# Dirichlet distribution
# =============================================================================


class DirichletDistribution(ExponentialFamilyDistribution):

    @property
    def torch_dist_class(self):
        return torch.distributions.Dirichlet

    @property
    def mean_params(self):
        raise NotImplementedError

    @staticmethod
    def _std_from_unc(unc_params):
        
        log_conc = unc_params["up1"]
        
        std = {
            "concentration": torch.exp(log_conc)
        }
        
        return std

    @staticmethod
    def _unc_from_std(std_params):
        
        conc = std_params["concentration"].detach()
        log_conc = torch.exp(conc)
        
        unc = {
            "up1": torch.nn.Parameter(log_conc)
        }
        
        return unc

    @staticmethod
    def _nat_from_std(std_params):
        
        conc = std_params["concentration"]
        
        nat = {
            "np1": conc - 1.
        }
        
        return nat

    @staticmethod
    def _std_from_nat(nat_params):
        
        conc_minus_one = nat_params["np1"]
        
        std = {
            "concentration": conc_minus_one + 1.
        }
        
        return std

    @staticmethod
    def _mean_from_std(std_params):
        raise NotImplementedError

    
# =============================================================================
# Multinomial distribution
# =============================================================================


class MultinomialDistribution(ExponentialFamilyDistribution):

    @property
    def mean_params(self):
        raise NotImplementedError

    @staticmethod
    def _std_from_unc(unc_params):
        
        # First parameter is the number of trials and therefore not learnable
        up1 = unc_params["up1"]
        up2 = unc_params["up2"]
        
        p = torch.exp(up2)
        p = p / p.sum()
        
        std = {
            "total_count": up1,
            "probs": p,
        }
        
        return std

    @staticmethod
    def _unc_from_std(std_params):
        
        # First parameter is the number of trials and therefore not learnable
        sp1 = std_params["total_count"]
        sp2 = std_params["probs"]
        
        n = sp1
        log_p = torch.log(sp2)
        
        unc = {
            "up1": torch.nn.Parameter(n, requires_grad=False).int(),
            "up2": torch.nn.Parameter(log_p),
        }
        
        return unc

    @staticmethod
    def _nat_from_std(std_params):
        
        # First parameter is the number of trials and therefore not learnable
        sp1 = std_params["total_count"]
        sp2 = std_params["probs"]
        
        log_p = torch.log(sp2)
        
        nat = {
            "np1": sp1,
            "np2": log_p,
        }
        
        return nat

    @staticmethod
    def _std_from_nat(nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        p = torch.exp(np2)
        p = p / p.sum()
        
        std = {
            "total_count": np1,
            "probs": p
        }

        return std

    @staticmethod
    def _mean_from_std(std_params):
        raise NotImplementedError

    @property
    def torch_dist_class(self):
        return torch.distributions.Multinomial


# =============================================================================
# Gamma distribution
# =============================================================================


class GammaDistribution(ExponentialFamilyDistribution):

    @property
    def mean_params(self):
        raise NotImplementedError

    @staticmethod
    def _std_from_unc(unc_params):
        log_alpha = unc_params["log_alpha"]
        log_beta = unc_params["log_beta"]

        concentration = log_alpha.exp()
        rate = 1 / log_beta.exp()

        std = {
            "concentration": concentration,
            "rate": rate
        }

        return std

    @staticmethod
    def _unc_from_std(std_params):
        concentration = std_params["concentration"].detach()
        rate = std_params["rate"].detach()

        unc = {
            "log_alpha": torch.nn.Parameter(concentration.log()),
            "log_beta": torch.nn.Parameter((1 / rate).log())
        }

        return unc

    @staticmethod
    def _nat_from_std(std_params):
        concentration = std_params["concentration"]
        rate = std_params["rate"]

        np1 = concentration - 1
        np2 = - 1 / rate

        nat = {
            "np1": np1,
            "np2": np2,
        }

        return nat

    @staticmethod
    def _std_from_nat(nat_params):
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]

        concentration = np1 + 1
        rate = -1 / np2

        std = {
            "concentration": concentration,
            "rate": rate
        }

        return std

    @staticmethod
    def _mean_from_std(std_params):
        raise NotImplementedError

    @property
    def torch_dist_class(self):
        return torch.distributions.Gamma


# =============================================================================
# Log-Normal distributions
# =============================================================================


class LogNormalDistribution(MeanFieldGaussianDistribution):

    @property
    def torch_dist_class(self):
        return torch.distributions.LogNormal
