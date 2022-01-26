import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from pvi.servers.base import Server
from pvi.utils.dataset import ListDataset
from pvi.utils.training_utils import EarlyStopping
from tqdm.auto import tqdm

# =============================================================================
# Global VI class
# =============================================================================


class GlobalVIServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tracks number of epochs.
        self.epochs = 0

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "train_model": False,
            "model_optimiser_params": {},
            "max_iterations": 1,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "early_stopping": EarlyStopping(np.inf),
            "num_elbo_samples": 10,
            "print_epochs": np.inf,
            "homogenous_split": False,
            "verbose": True,
        }

    def tick(self):
        """
        Trains the model using the entire dataset.
        """
        if self.should_stop():
            return False

        p = self.p.non_trainable_copy()

        if self.iterations == 0:
            if self.init_q is not None:
                q = self.init_q.trainable_copy()
            else:
                q = self.p.trainable_copy()
        else:
            q = self.q.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            parameters = [
                {"params": q.parameters()},
                {
                    "params": self.model.parameters(),
                    **self.config["model_optimiser_params"],
                },
            ]
        else:
            parameters = q.parameters()

        # Reset optimiser.
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )

        # Set up data: global VI server can access the entire dataset.
        if self.config["homogenous_split"]:
            # Set up data loader.
            tensor_dataset = TensorDataset(*self.data.values())
            loader = DataLoader(
                tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
            )
        else:
            # Inhomogenous split: order matters.
            data = defaultdict(list)
            for client in self.clients:
                # Chunk clients data into batch size.
                data = {
                    k: data[k]
                    + [
                        v[i : i + self.config["batch_size"]]
                        for i in range(
                            0, len(client.data["x"]), self.config["batch_size"]
                        )
                    ]
                    for k, v in client.data.items()
                }

            # Lists of tensors size (batch_size, *).
            list_dataset = ListDataset(*data.values())
            loader = DataLoader(list_dataset, batch_size=None, shuffle=True)

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](scores=None, model=q.non_trainable_copy())

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(
            range(self.config["epochs"]),
            desc="Epochs",
            disable=(not self.config["verbose"]),
        )
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)
            for batch in iter(loader):
                batch = {
                    k: batch[i].to(self.config["device"])
                    for i, k in enumerate(self.data.keys())
                }
                self.communications += 1
                optimiser.zero_grad()

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(p).sum() / len(self.data["x"])

                # Compute E_q[log p(y | x, θ)].
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]
                ).sum()
                ll /= len(batch["x"])

                elbo = ll - kl
                loss = -elbo
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += elbo.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(epoch)

            # Log progress for current epoch.
            for k, v in epoch.items():
                training_metrics[k].append(v)

            stop_early = self.config["early_stopping"](
                training_metrics, model=q.non_trainable_copy()
            )

            if (
                i % self.config["print_epochs"] == 0
                or i == (self.config["epochs"] - 1)
                or stop_early
            ):
                # Update global posterior before evaluating performance.
                self.q = q.non_trainable_copy()

                metrics = self.evaluate_performance({"epochs": i, **epoch})

                if self.config["verbose"]:
                    # Report performance.
                    report = ""
                    report += f"epochs: {metrics['epochs']} "
                    report += f"elbo: {metrics['elbo']:.3f} "
                    report += f"ll: {metrics['ll']:.3f} "
                    report += f"kl: {metrics['kl']:.3f} \n"
                    for k, v in metrics.items():
                        if "mll" in k or "acc" in k:
                            report += f"{k}: {v:.3f} "

                    tqdm.write(report)

            # Log q so we can track performance each epoch.
            self.log["communications"].append(self.communications)
            self.epochs += 1

            # Check whether to stop early.
            if stop_early:
                break

        # Create non-trainable copy to send back to server.
        if self.config["early_stopping"].stash_model:
            self.q = self.config["early_stopping"].best_model
        else:
            self.q = q.non_trainable_copy()

        self.iterations += 1

        # Log training.
        self.log["training_curves"].append(training_metrics)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
