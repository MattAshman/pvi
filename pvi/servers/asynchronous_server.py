import logging
import time
import numpy as np
import ray

from tqdm.auto import tqdm
from pvi.servers.base import Server
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


@ray.remote(num_gpus=.1, num_cpus=.1)
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


@ray.remote(num_gpus=.1, num_cpus=.1)
class RayClient(Client):

    def get_log(self):
        return self.log

    def update_client(self, q, init_q=None):
        print(f"Updating client {str(id(self))[-3:]}.")

        t_old = self.t
        _, t_new = self.fit(q, init_q)

        return t_old, t_new


class AsynchronousRayFactory(Server):
    """
    This acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
        }

    def tick(self):

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        working_clients = []
        for i, client in enumerate(self.clients):
            working_clients.append(update_client.remote(
                client, self.q, self.init_q, i))

        while not self.should_stop():
            ready_clients, _ = ray.wait(list(working_clients))
            client_id = ready_clients[0]
            client_idx = working_clients.index(client_id)

            # Apply change in factors.
            self.clients[client_idx], t_old, t_new = ray.get(client_id)
            self.q = update_q.remote(self.q, *[client_id])

            # Get client training again.
            if self.config["init_q_always"]:
                working_clients[client_idx] = update_client.remote(
                    self.clients[client_idx], self.q, self.init_q,
                    client_idx=client_idx)
            else:
                working_clients[client_idx] = update_client.remote(
                    self.clients[client_idx], self.q, client_idx=client_idx)

            self.communications += 1
            if self.communications % len(self.clients) == 0:
                # Evaluate current posterior.
                # Can we make evauation a remote function too?
                self.q = ray.get(self.q)
                self.evaluate_performance()
                self.log["communications"].append(self.communications)

                metrics = self.log["performance_metrics"][-1]
                print("Communications: {}.".format(self.communications))
                print("Test mll: {:.3f}. Test acc: {:.3f}.".format(
                    metrics["val_mll"], metrics["val_acc"]))
                print("Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
                    metrics["train_mll"], metrics["train_acc"]))

    def should_stop(self):
        com_test = self.communications > self.config["max_communications"] - 1

        if len(self.log["performance_metrics"]) > 0:
            perf_test = self.log["performance_metrics"][-1]["val_mll"] < -10
        else:
            perf_test = False

        return com_test or perf_test


class SynchronousRayFactory(Server):
    """
    This acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
        }

    def tick(self):

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        working_clients = []
        for i, client in enumerate(self.clients):
            working_clients.append(update_client.remote(
                client, self.q, self.init_q, i))

        while not self.should_stop():
            # Pass current q to clients.
            if self.iterations == 0 or self.config["init_q_always"]:
                working_clients = [
                    update_client.remote(client, self.q, self.init_q, i)
                    for i, client in enumerate(self.clients)]
            else:
                working_clients = [
                    update_client.remote(client, self.q, client_idx=i)
                    for i, client in enumerate(self.clients)]

            # Apply change in factors.
            self.q = update_q.remote(self.q, *working_clients)
            self.communications += len(self.clients)
            self.iterations += 1

            # Evaluate current posterior.
            self.q = ray.get(self.q)
            self.evaluate_performance()
            self.log["communications"].append(self.communications)

            metrics = self.log["performance_metrics"][-1]
            print("Communications: {}.".format(self.communications))
            print("Test mll: {:.3f}. Test acc: {:.3f}.".format(
                metrics["val_mll"], metrics["val_acc"]))
            print("Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
                metrics["train_mll"], metrics["train_acc"]))

    def should_stop(self):
        iter_test = self.iterations > self.config["max_iterations"] - 1

        if len(self.log["performance_metrics"]) > 0:
            perf_test = self.log["performance_metrics"][-1]["val_mll"] < -10
        else:
            perf_test = False

        return iter_test or perf_test


@ray.remote(num_gpus=.1, num_cpus=.1)
def update_client(client, q, init_q=None, client_idx=None):
    if client_idx is not None:
        print(f"Updating client {client_idx}.")

    t_old = client.t
    _, t_new = client.fit(q, init_q)

    return client, t_old, t_new


@ray.remote(num_gpus=.1, num_cpus=.1)
def update_q(q, *ts):
    for t in ts:
        _, t_old, t_new = t

        # Update posterior.
        q = q.replace_factor(t_old, t_new, is_trainable=False)

    return q
