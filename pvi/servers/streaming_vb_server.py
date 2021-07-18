import logging

from tqdm.auto import tqdm
from .base import *

logger = logging.getLogger(__name__)


class StreamingVBServer(Server):

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
            "shared_factor_iterations": 0,
            "init_q_to_all": False
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                # In streaming VB, we set t(Θ) to 1 so data is recounted.
                # Equivalent to sequential PVI with no deletion.
                client.t.nat_params = {k: torch.zeros_like(v)
                                       for k, v in client.t.nat_params}

                t_old = client.t

                if (not self.config["init_q_to_all"] and self.communications
                    == 0) or (self.config["init_q_to_all"] and
                              self.iterations == 0):
                    # First iteration. Pass q_init(θ) to client.
                    _, t_new = client.fit(self.q, self.init_q)
                else:
                    _, t_new = client.fit(self.q)

                if self.iterations < self.config["shared_factor_iterations"]:
                    # Update shared factor.
                    n = len(self.clients)
                    t_np = {k: (v * (1 - 1 / n) +
                                t_new.nat_params[k] * (1 / n))
                            for k, v in t_old.nat_params.items()}

                    # Set clients factor to shared factor.
                    for j in range(len(self.clients)):
                        self.clients[j].t.nat_params = t_np

                # Only update global posterior.
                self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)

                logger.debug(
                    "Received client update. Updating global posterior.")

                self.communications += 1

                # Evaluate performance after every posterior update.
                self.evaluate_performance()
                self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete.\n")

        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
