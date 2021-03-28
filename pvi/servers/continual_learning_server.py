import logging

from .base import Server, BayesianServer

logger = logging.getLogger(__name__)


class ContinualLearningServer(Server):
    def __init__(self, model, q, clients, hyperparameters=None):
        super().__init__(model, q, clients, hyperparameters)

        self.client_idx = 0
        self.log["q"].append(self.q)

    def get_default_hyperparameters(self):
        return {
            **super().get_default_hyperparameters(),
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        client = self.clients[self.client_idx]

        if client.can_update():
            # TODO: ensure that client.fit returns non-trainable copy?
            q_new = client.fit(self.q)
            self.q = q_new.non_trainable_copy()

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1
        self.client_idx = (self.client_idx + 1) % len(self.clients)

        # Log progress.
        self.log["q"].append(self.q)
        self.log["communications"].append(1)

    def should_stop(self):
        return self.iterations > self.hyperparameters["max_iterations"] - 1


class BayesianContinualLearningServer(BayesianServer):
    def __init__(self, model, q, qeps, clients, hyperparameters=None):
        super().__init__(model, q, qeps, clients, hyperparameters)

        self.client_idx = 0
        self.log["q"].append(self.q)

    def get_default_hyperparameters(self):
        return {
            **super().get_default_hyperparameters(),
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        client = self.clients[self.client_idx]

        if client.can_update():
            # TODO: ensure that client.fit returns non-trainable copy?
            q_new, qeps_new = client.fit(self.q, self.qeps)
            self.q = q_new.non_trainable_copy()
            self.qeps = qeps_new.non_trainable_copy()

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1
        self.client_idx = (self.client_idx + 1) % len(self.clients)

        # Log progress.
        self.log["q"].append(self.q)
        self.log["qes"].append(self.qeps)
        self.log["communications"].append(1)

    def should_stop(self):
        return self.iterations > self.hyperparameters["max_iterations"] - 1
