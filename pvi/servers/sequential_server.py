import logging

from tqdm.auto import tqdm
from .base import *

logger = logging.getLogger(__name__)


class SequentialServer(Server):

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

                # Evaluate performance after every posterior update for first
                # iteration.
                if self.iterations == 0:
                    self.evaluate_performance()
                    self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete.\n")

        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

        # Evaluate performance after every iteration.
        self.evaluate_performance()
        self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class SequentialServerBayesianHypers(ServerBayesianHypers):

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 5,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_old = client.t
                teps_old = client.teps

                if self.iterations == 0:
                    _, _, t_new, teps_new = client.fit(
                        self.q, self.qeps, self.init_q, self.init_qeps)
                else:
                    _, _, t_new, teps_new = client.fit(self.q, self.qeps)

                logger.debug(
                    "Received client update. Updating global posterior.")

                self.q = self.q.replace_factor(t_old, t_new,
                                               is_trianable=False)
                self.qeps = self.qeps.replace_factor(teps_old, teps_new,
                                                     is_trainable=False)

                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                # self.log["q"].append(self.q.non_trainable_copy())
                # self.log["qeps"].append(self.qeps.non_trainable_copy())
                self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete.\n")

        self.iterations += 1

        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
