from base import Client


# =============================================================================
# Synchronous client class
# =============================================================================


class SynchronousClient(Client):
    
    def __init__(self, data, model, t):
        super().__init__(data=data, model=model, t=t)
        
        
    def fit(self, q):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called directly by the server.
        """
        
        # Compute new posterior (ignored) and approximating likelihood term
        _, self.t = super().update_q(q, self.t)
        
        return self.t