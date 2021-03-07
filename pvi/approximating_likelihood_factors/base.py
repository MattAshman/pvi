from abc import ABC, abstract_method


# =============================================================================
# Base approximating likelihood class
# =============================================================================


class ApproximatingLikelihoodFactor(ABC):
    
    # TODO:
    # - compute_refined_factor: computes refined factor t
    # - deal with the initialisation t(θ) = 1
    
    def __init__(self, log_coefficient, natural_parameters):
        
        # Set leading coefficient and natural parameters
        self.log_coefficient = log_coefficient
        self.natural_parameters = natural_parameters
    
    
    @abstractmethod
    def __call__(self, thetas):
        """
        Computes the log-value of θ under the factor t, i.e. log t(θ).
        """
        pass
    
    
    @abstractmethod
    def compute_refined_factor(self, q, q_):
        """
        Computes the natural parameters and leading coefficient of the
        approximating likelihood term **t** given by
        
            t(θ) = q(θ) / q_(θ) t_(θ)
            
        where **t_** is the approximating likelihood term corresponding
        to **self**.
        """
        pass
        
            