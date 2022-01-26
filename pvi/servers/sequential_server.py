from tqdm.auto import tqdm
from pvi.servers.base import Server, ServerBayesianHypers


class SequentialServer(Server):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
        }

    def _tick(self):
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                t_old = client.t

                if self.communications == 0:
                    # First iteration. Pass q_init(θ) to client.
                    _, t_new = client.fit(self.q, self.init_q)
                else:
                    _, t_new = client.fit(self.q)

                self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)
                self.communications += 1

                # Evaluate performance after every posterior update for first iteration.
                if self.iterations == 0:
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

    def _tick(self):
        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                t_old = client.t
                teps_old = client.teps

                if self.iterations == 0:
                    _, _, t_new, teps_new = client.fit(
                        self.q, self.qeps, self.init_q, self.init_qeps
                    )
                else:
                    _, _, t_new, teps_new = client.fit(self.q, self.qeps)

                self.q = self.q.replace_factor(t_old, t_new, is_trianable=False)
                self.qeps = self.qeps.replace_factor(
                    teps_old, teps_new, is_trainable=False
                )

                clients_updated += 1
                self.communications += 1
                self.log["communications"].append(self.communications)

        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
