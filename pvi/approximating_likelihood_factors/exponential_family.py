from abc import ABC, abstract_method
from base import ApproximatingLikelihoodFactor


# =============================================================================
# Base class for approximating likelihood factors of the exponential family
# =============================================================================


def ExponentialFamilyFactor(ApproximatingLikelihoodFactor):
    """
    Base class for exponential family (EF) approximating likelihood factors.
    The exponential family is made up of distributions which can be written
    in the form
    
        p(x | θ) = h(x) exp(ν.T f(x) + A(ν)).
    
    For a list of exponential family distributions see:
        https://en.wikipedia.org/wiki/Category:Exponential_family_distributions
        https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
    
    
    TODO: Write children classes for the following members of the EF
    
        - MultivariateGaussian
        - Laplace
        - Dirichlet
        
    There are many more members but these are especially useful for our
    applications.
    """
    
    def __init__(self, log_coefficient, natural_parameters):
        
        super().__init__(log_coefficient=log_coefficient,
                         natural_parameters=natural_parameters)
      
    
    def compute_refined_factor(self, q1, q2):
        """
        Computes the log-coefficient and natural parameters of the
        approximating likelihood term **t** given by
        
            t(θ) = q1(θ) / q2(θ) t_(θ)
            
        where **t_** is the approximating likelihood term corresponding
        to **self**. Note that the log-coefficient computed here includes
        the normalising constants of the q-distributions as well as the
        coefficient of t_.
        """
        
        # Convert distributions to log-coefficients and natural parameters
        log_coeff1, np1 = self.log_coeff_and_np_from_distribution(q1)
        log_coeff1, np2 = self.log_coeff_and_np_from_distribution(q2)
        
        # Log coefficient and natural parameters of refined factor
        log_coeff = log_coeff1 - log_coeff2 + self.log_coeff
        natural_parameters = {}
        
        # Compute natural parameters of the new t-factor
        for name, np in self.natural_parameters.items():
            natural_parameters[name] = np1[name] - np2[name] + np
            
        # Create and return refined t
        t = type(self).__init__(log_coeff, natural_parameters)
        
        return t
    

    @abstractmethod
    def __call__(self, thetas):
        """
        Returns the value of log t(θ). Input **thetas** is assumed to be
        a torch.tensor of shape (N, D) where N is the batch dimension and
        D is the number of natural parameters of the distribution.
        """
        pass
    
    
    @abstractmethod
    def log_h(self, thetas):
        """
        Returns the value of log h(θ). Input **thetas** is assumed to be
        a torch.tensor of shape (N, D) where N is the batch dimension and
        D is the dimension of the distribution.
        """
        pass
    
    
    @abstractmethod
    def log_coeff_and_np_from_distribution(self, q):
        """
        Takes a torch.distribution **q**, assumed to be in the EF
        and extracts its leading coefficient and natural parameters.
        """
        pass
    
    
    @abstractmethod
    def f(self, thetas):
        """
        Computes f(θ) for this member of the exponential family. 
        """
        pass
    


# =============================================================================
# Mean field Gaussian factor
# =============================================================================
    

class MeanFieldGaussian(ExponentialFamilyFactor):
    
    
    def __init__(self, log_coeff, natural_parameters):
        
        super().__init__(log_coefficient, natural_parameters)
    
    
    def __call__(self, thetas):
        """
        Returns the value of log t(θ) where
        
            log t(θ) = log c + log h(x) + ν.T f(x)
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        
        # Extract NP tensors, compute f(θ) tensors, list [shape (N, D)]
        np = self.natural_parameters.values()
        f = self.f(thetas)
        
        # Compute log h(θ), shape (N,)
        log_h = self.log(h)
        
        # Compute ν.T f(x), shape (N,)
        npf = [torch.einsum('ni, ni -> n', np_, f_) for np_, f_ in zip(np, f)]
        npf = torch.sum(torch.stack(npf, dim=1), dim=1)
        
        # Compute log t(θ), shape (N,)
        log_t = self.log_coefficient + log_h + npf
        
        return log_t
    
    
    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the MeanFieldFactor class. For
        a mean-field multivariate Gaussian, log h(θ) = 0.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        
        log_h = torch.zeros(size=thetas.shape[0])
        
        return log_h
    
    
    def f(self, thetas):
        """
        Computes f(θ) for the mean-field Gaussian where
        
            f(θ) = [θΤ, θΤ ** 2]Τ
        """
        
        return [thetas, thetas ** 2]
    
    
    def log_coeff_and_np_from_distribution(self, q):
        """
        Extracts the coefficient and natural parameters of **q**, where
        **q** is assumed to be a torch.distribution.
        """
        
        loc = q.loc.item().detach()
        scale = q.scale.item().detach()
        
        D = mean.shape[0]
        
        log_coeff = torch.log(scale).sum()
        log_coeff = log_coeff - 0.5 * D * np.log(2 * np.pi)
        
        np = {
            "np1" : loc / scale ** 2,
            "np2" : - 0.5 * scale ** -2
        }
        
        return log_coeff, np
