from abc import ABC, abstractmethod
from collections import defaultdict


class Server(ABC):
    """
    An abstract class for the server.
    """
    def __init__(self, model, q, clients, config=None):

        if config is None:
            config = {}

        self._config = self.get_default_config()
        self.config = config

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

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    @abstractmethod
    def get_default_config(self):
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
        return self.model(x, self.q)

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


class BayesianServer(Server):
    def __init__(self, model, q, qeps, clients, config=None):
        super().__init__(model, q, clients, config)

        # Global posterior q(ε).
        self.qeps = qeps

    @abstractmethod
    def get_default_config(self):
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
        :return: ∫ p(y | x, θ, ε) q(θ)q(ε) dθ dε.
        """
        neps = self.config["num_eps_samples"]
        dists = []
        for _ in range(neps):
            eps = self.qeps.sample()
            self.model.hyperparameters = eps
            dists.append(self.model(x, self.q))

        return dists
