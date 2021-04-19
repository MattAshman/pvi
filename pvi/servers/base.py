from abc import ABC, abstractmethod
from collections import defaultdict


class Server(ABC):
    """
    An abstract class for the server.
    """
    def __init__(self, model, q, clients, hyperparameters=None):

        if hyperparameters is None:
            hyperparameters = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.set_hyperparameters(hyperparameters)

        # Shared probabilistic model.
        self.model = model

        # Global posterior q(θ).
        self.q = q

        # Clients.
        self.clients = clients

        # Internal iteration counter.
        self.iterations = 0

        # Internal communication counter.
        self.communications = 0

        self.log = defaultdict(list)

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    @abstractmethod
    def get_default_hyperparameters(self):
        return {}

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
        return self.model(x, self.q, model_predict=True)

    def add_client(self, client):
        self.clients.append(client)

    def get_compiled_log(self):
        """
        Get full log, including logs from each client.
        :return: full log.
        """
        final_log = {
            "server": self.log
        }

        client_logs = [client.log for client in self.clients]
        for i, log in enumerate(client_logs):
            final_log["client_" + str(i)] = log

        return final_log
