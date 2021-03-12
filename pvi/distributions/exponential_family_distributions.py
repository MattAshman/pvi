from .base import ExponentialFamilyDistribution

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
        
        
    def _std_from_unc(self, unc_params):
        
        loc = unc_params["up1"]
        log_scale = unc_params["up2"]
        
        std = {
            "loc" : loc,
            "scale" : torch.exp(log_scale)
        }
        
        return std
    
    
    def _unc_from_std(self, std_params):
        
        loc = std_params["loc"].detach()
        scale = std_params["scale"].detach()
        
        unc = {
            "up1" : torch.nn.Parameter(loc),
            "up2" : torch.nn.Parameter(torch.log(scale))
        }
        
        return unc
    
    
    @classmethod
    def _nat_from_std(self, std_params):
        
        loc = std_params["loc"]
        scale = std_params["scale"]
        
        nat = {
            "np1" : loc * scale ** -2,
            "np2" : -0.5 * scale ** -2
        }
        
        return nat
    
    
    @classmethod
    def _std_from_nat(self, nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        np = {
            "loc" : - 0.5 * np1 / np2,
            "scale" : (- 0.5 / np2) ** 0.5
        }
        
        return np

    
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
        
        
    def _std_from_unc(self, unc_params):
        
        loc = unc_params["up1"]
        scale_tril = unc_params["up2"]
        
        std = {
            "loc" : loc,
            "covariance_matrix" : torch.mm(scale_tril, scale_tril.T)
        }
        
        return std
    
    
    def _unc_from_std(self, std_params):
        
        loc = std_params["loc"].detach()
        cov = std_params["covariance_matrix"].detach()
        
        scale_tril = torch.linalg.cholesky(cov)
        
        unc = {
            "up1" : torch.nn.Parameter(loc),
            "up2" : torch.nn.Parameter(scale_tril)
        }
        
        return unc
    
    
    @classmethod
    def _nat_from_std(self, std_params):
        
        loc = std_params["sp1"]
        cov = std_params["sp2"]
        
        nat = {
            "np1" : torch.solve(loc[:, None], cov).solution[:, 0],
            "np2" : -0.5 * torch.inverse(cov)
        }
        
        return nat
    
    
    @classmethod
    def _std_from_nat(self, nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        prec = -2. * np2
        
        np = {
            "loc" : torch.solve(np1[:, None], prec).solution[:, 0],
            "covariance_matrix" : torch.inverse(prec)
        }
        
        return np

    
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
        pass
        
    
    @property
    def torch_dist_class(self):
        return torch.distributions.Dirichlet
        
        
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
    def _nat_from_std(self, std_params):
        
        conc = std_params["concentration"]
        
        nat = {
            "np1" : conc - 1.
        }
        
        return nat
    
    
    @classmethod
    def _std_from_nat(self, nat_params):
        
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
        
        return std
    
    
    @classmethod
    def _nat_from_std(self, std_params):
        
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
    def _std_from_nat(self, nat_params):
        
        np1 = nat_params["np1"]
        np2 = nat_params["np2"]
        
        p = torch.exp(np2)
        p = p / p.sum()
        
        std = {
            "total_count" : np1,
            "probs" : p
        }
        
    
    @property
    def torch_dist_class(self):
        return torch.distributions.Multinomial
    