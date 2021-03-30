from .base import PVIClient


# =============================================================================
# Synchronous client class
# =============================================================================


class SynchronousClient(PVIClient):
    
    def __init__(self, data, model, t):
        super().__init__(data=data, model=model, t=t)

    def fit(self, q):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called directly by the server.
        """
        
        # Compute new posterior (ignored) and approximating likelihood term
        _, self.t = super().update_q(q)
        
        return self.t
