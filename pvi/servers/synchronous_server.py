import logging

from tqdm.auto import tqdm
from .base import Server

logger = logging.getLogger(__name__)


class SynchronousServer(Server):
    def __init__(self, model, q, clients, config=None):
        super().__init__(model, q, clients, config)

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 5,
            "damping_factor": 1.,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        damping = self.config["damping_factor"]
        delta_nps = []
        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                t_i_new = client.fit(self.q)
                # Compute change in natural parameters.
                delta_np = {}
                for k in self.q.nat_params.keys():
                    delta_np[k] = t_i_new.nat_params[k] - t_i_old.nat_params[k]

                delta_nps.append(delta_np)
                clients_updated += 1
                self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        q_new_nps = {}
        for k, v in self.q.nat_params.items():
            q_new_nps[k] = (
                    v + sum([delta_np[k] for delta_np in delta_nps]) * damping)

        self.q = type(self.q)(nat_params=q_new_nps, is_trainable=False)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)
        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
