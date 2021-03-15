from abc import ABC, abstractmethod
from collections import defaultdict


class Server(ABC):
    """
    An abstract class for the server.
    """
    def __init__(self, model, q, clients, max_iterations):
        # Shared probabilistic model.
        self.model = model

        # Global posterior q(θ).
        self.q = q

        # Clients.
        self.clients = clients

        # Maximum number of iterations before stopping.
        self.max_iterations = max_iterations
        self.iterations = 0

        self.log = defaultdict(list)

    @abstractmethod
    def tick(self):
        """
        Defines what the server should do on each update round. Could be a
        synchronous update, asynchronous update etc.
        """
        pass

    @abstractmethod
    def should_stop(self):
        """
        Defines when the server should stop running.
        """
        pass

    def model_predict(self, x):
        """
        Returns the current models predictive posterior distribution.
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        return self.model(x, self.q)

    def add_client(self, client):
        self.clients.append(client)
