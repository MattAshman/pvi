import logging

from tqdm.auto import tqdm
from .base import Server

logger = logging.getLogger(__name__)


class SequentialServer(Server):
    def __init__(self, model, q, clients, hyperparameters=None):
        super().__init__(model, q, clients, hyperparameters)

    def get_default_hyperparameters(self):
        return {
            **super().get_default_hyperparameters(),
            "max_iterations": 100,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                t_i_new = client.fit(self.q)
                # Compute change in natural parameters.
                delta_np = {k : (t_i_new.nat_params[k] - t_i_old.nat_params[k])
                            for k in self.q.nat_params.keys()}

                logger.debug(
                    "Received client update. Updating global posterior.")
                # Update global posterior.
                q_new_nps = {k : v + delta_np[k]
                             for k, v in self.q.nat_params.items()}

                self.q = type(self.q)(nat_params=q_new_nps, is_trainable=False)
                clients_updated += 1

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.log["nat_params"].append(self.q.nat_params)

    def should_stop(self):
        return self.iterations > self.hyperparameters["max_iterations"] - 1
