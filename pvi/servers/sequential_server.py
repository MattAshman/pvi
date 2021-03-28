import logging

from tqdm.auto import tqdm
from .base import Server

logger = logging.getLogger(__name__)


class SequentialServer(Server):
    def __init__(self, model, q, clients, hyperparameters=None):
        super().__init__(model, q, clients, hyperparameters)

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_hyperparameters(self):
        return {
            **super().get_default_hyperparameters(),
            "max_iterations": 20,
            "damping_factor": 1.,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        damping = self.hyperparameters["damping_factor"]
        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                t_i_new = client.fit(self.q)
                # Compute change in natural parameters.
                delta_np = {k: (t_i_new.nat_params[k] - t_i_old.nat_params[k])
                            for k in self.q.nat_params.keys()}

                logger.debug(
                    "Received client update. Updating global posterior.")
                # Update global posterior.
                q_new_nps = {k: v + delta_np[k] * damping
                             for k, v in self.q.nat_params.items()}

                self.q = type(self.q)(nat_params=q_new_nps, is_trainable=False)
                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                self.log["q"].append(self.q.non_trainable_copy())
                self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.hyperparameters["max_iterations"] - 1
