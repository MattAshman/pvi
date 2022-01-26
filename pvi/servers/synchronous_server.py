from tqdm.auto import tqdm
from pvi.servers.base import Server, ServerBayesianHypers


class SynchronousServer(Server):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
        }

    def _tick(self):
        t_olds = []
        t_news = []
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                t_old = client.t

                if self.iterations == 0:
                    # First iteration. Pass q_init(θ) to client.
                    _, t_new = client.fit(self.q, self.init_q)
                else:
                    _, t_new = client.fit(self.q)

                t_olds.append(t_old)
                t_news.append(t_new)

        # Single communication per iteration.
        self.communications += 1

        # Update global posterior.
        for t_old, t_new in zip(t_olds, t_news):
            self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class SynchronousServerBayesianHypers(ServerBayesianHypers):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 5,
        }

    def _tick(self):
        clients_updated = 0

        t_olds, t_news = [], []
        teps_olds, teps_news = [], []
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

                t_olds.append(t_old)
                t_news.append(t_new)
                teps_olds.append(teps_old)
                teps_news.append(teps_new)

                clients_updated += 1
                self.communications += 1

        # Update global posterior.
        for t_old, t_new, teps_old, teps_new in zip(
            t_olds, t_news, teps_olds, teps_news
        ):
            self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)
            self.qeps = self.qeps.replace_factor(teps_old, teps_new, is_trainable=False)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
