from base import Client


# =============================================================================
# Synchronous client class
# =============================================================================


class SynchronousClient(Client):
    
    def __init__(self, data, likelihood, t):
        super().__init__(data=data, likelihood=likelihood, t=t)
        
        
    def fit(self, q):
        """ Computes a refined posterior and its associated approximating
        likelihood term. This method is called directly by the server.
        """
        
        # Compute new posterior (ignored) and approximating likelihood term
        _, self.t = super().q_update(q, self.t)
        
        return self.t