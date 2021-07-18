import logging
import numpy as np
import ray

from tqdm.auto import tqdm
from .base import *
from pvi.clients.base import Client

logger = logging.getLogger(__name__)


class SynchronousRayServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.server_worker = ServerWorker.remote(self.q)

    def tick(self):
        q = self.server_worker.get_current_q.remote()

        while not self.should_stop():
            # Pass current q to clients.
            working_clients = [
                    client.update_client.remote(q, self.init_q)
                    for client in self.clients
            ]

            # Apply change in factors.
            q = self.server_worker.update_q.remote(*working_clients)
            self.communications += len(self.clients)
            self.iterations += 1

            # Evaluate current posterior.
            self.q = ray.get(q)
            self.evaluate_performance()
            self.log["communications"].append(self.communications)

            metrics = self.log["performance_metrics"][-1]
            print("Communications: {}.".format(self.communications))
            print("Test mll: {:.3f}. Test acc: {:.3f}.".format(
                metrics["val_mll"], metrics["val_acc"]))
            print("Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
                metrics["train_mll"], metrics["train_acc"]))

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class AsynchronousRayServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.server_worker = ServerWorker.remote(self.q)

    def tick(self):
        working_clients = {}
        q = self.server_worker.get_current_q.remote()
        for client in self.clients:
            working_clients[
                    client.update_client.remote(q, self.init_q)] = client

        while not self.should_stop():
            ready_clients, _ = ray.wait(list(working_clients))
            ready_client_id = ready_clients[0]
            client = working_clients.pop(ready_client_id)

            # Apply change in factors.
            q = self.server_worker.update_q.remote(*[ready_client_id])

            # Get client training again.
            working_clients[client.update_client.remote(q)] = client

            self.communications += 1
            if self.communications % len(self.clients) == 0:
                # Evaluate current posterior.
                self.q = ray.get(q)
                self.evaluate_performance()
                self.log["communications"].append(self.communications)

                metrics = self.log["performance_metrics"][-1]
                print("Communications: {}.".format(self.communications))
                print("Test mll: {:.3f}. Test acc: {:.3f}.".format(
                    metrics["val_mll"], metrics["val_acc"]))
                print("Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
                    metrics["train_mll"], metrics["train_acc"]))

    def should_stop(self):
        return self.communications > self.config["max_communications"] - 1


@ray.remote
class ServerWorker():

    def __init__(self, q):
        self.q = q

    def update_q(self, *ts):
        for t in ts:
            t_old, t_new = t

            # Update global posterior.
            self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)

        return self.q

    def get_current_q(self):
        return self.q


@ray.remote
class RayClient(Client):

    def update_client(self, q, init_q=None):
        print(f"Updating client {id(self)}.")

        t_old = self.t
        _, t_new = self.fit(q, init_q)

        return t_old, t_new

