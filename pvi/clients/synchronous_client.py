from .base import PVIClient, PVIClientBayesianHypers


# =============================================================================
# Synchronous client class
# =============================================================================


class SynchronousClient(PVIClient):
    
    def __init__(self, data, model, t, config=None):
        super().__init__(data=data, model=model, t=t, config=config)

    def fit(self, q):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called directly by the server.
        """
        
        # Compute new posterior (ignored) and approximating likelihood term
        _, self.t = super().update_q(q)
        
        return self.t


class SynchronousClientBayesianHypers(PVIClientBayesianHypers):

    def __init__(self, data, model, t, teps, config=None):
        super().__init__(data=data, model=model, t=t, teps=teps, config=config)

    def fit(self, q, qeps):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called directly by the server.
        """

        # Compute new posterior (ignored) and approximating likelihood term
        _, _, self.t, self.teps = super().update_q(q, qeps)

        return self.t, self.teps
