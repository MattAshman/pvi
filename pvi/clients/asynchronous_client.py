from abc import ABC, abstractmethod


# =============================================================================
# Asynchronous client class
# =============================================================================


class AsynchronousClient(ABC):
    
    def __init__(self, server, client_id, data, likelihood, t):
        
        # Set server and the client id
        self.server = server
        self.client_id = client_id
        
        # Set data partition and likelihood
        self.data = data
        self.likelihood = likelihood
        
        # Set likelihood approximating term
        self.t = t
    
    
    def fit(self):
        """
        Perpetually queries the server for the latest q and updates it.
        
        TODO:
        
            - Add more sophisticated scheduling, e.g. maximum number of updates
        or maximum number of updates per unit time etc.
            - Add a step which checks if the q has been updated since the last
            time the server was queried, to avoid excess computations.
        """
        while True: self._fit()
        
        
    def _fit(self):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called within AsynchronousClient.fit.
        """
        
        # Retrieve newest q from server
        q = self.server.q
        
        # Compute new posterior (ignored) and approximating likelihood term
        self.q, self.t = super().q_update(q, self.t)
        
        # Send new appeoximating likelihood term to server
        self.server.update_posterior(self.t, client_id)
        
        return self.t