import logging

from .base import Server

logger = logging.getLogger(__name__)


class StreamingSGPServer(Server):
    def __init__(self, model, q, clients, hyperparameters=None):
        super().__init__(model, q, clients, hyperparameters)

    def get_default_hyperparameters(self):
        return {
            **super().get_default_hyperparameters(),
            "max_iterations": 10,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        client = self.clients[self.iterations]

        if client.can_update():
            self.q = client.fit(self.q)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.log["nat_params"].append(self.q.nat_params)
        self.log["inducing_locations"].append(self.q.inducing_locations)

    def should_stop(self):
        if self.iterations > self.hyperparameters["max_iterations"] - 1:
            return True
        else:
            return False
