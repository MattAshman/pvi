import logging
import torch

from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from .base import Server
from pvi.utils.dataset import ListDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# Global VI class
# =============================================================================


class GlobalVIServer(Server):
    def __init__(self, model, p, clients, config=None, init_q=None,
                 val_data=None):
        super().__init__(model, p, clients, config, init_q)

        # Global VI server has access to the entire dataset.
        self.data = {k: torch.cat([client.data[k] for client in self.clients],
                                  dim=0)
                     for k in self.clients[0].data.keys()}

        # Tracks number of epochs.
        self.epochs = 0

        # Validation dataset.
        self.val_data = val_data

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "train_model": False,
            "model_optimiser_params": {"lr": 1e-2},
            "max_iterations": 1,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "lr_scheduler": "MultiplicativeLR",
            "lr_scheduler_params": {
                "lr_lambda": lambda epoch: 1.
            },
            "num_elbo_samples": 10,
            "print_epochs": 1,
            "homogenous_split": True,
        }

    def tick(self):
        """
        Trains the model using the entire dataset.
        """
        if self.should_stop():
            return False

        p = self.p.non_trainable_copy()

        if self.iterations == 0 and self.init_q is not None:
            q = self.init_q.traiable_copy()
        else:
            q = self.q.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            parameters = [
                {"params": q.parameters()},
                {"params": self.model.parameters(),
                 **self.config["model_optimiser_params"]}
            ]
        else:
            parameters = q.parameters()

        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"])

        # Set up data: global VI server can access the entire dataset.
        if self.config["homogenous_split"]:
            x = self.data["x"]
            y = self.data["y"]

            # Shuffle data.
            tensor_dataset = TensorDataset(x, y)
            loader = DataLoader(tensor_dataset,
                                batch_size=self.config["batch_size"],
                                shuffle=True)
        else:
            # Inhomogenous split: order matters.
            m = self.config["batch_size"]
            data = defaultdict(list)
            for client in self.clients:
                # Chunk clients data into batch size.
                data = {k: data[k] + [v[i:i + m] for i in range(
                    0, len(client.data["x"]), m)]
                        for k, v in client.data.items()}

            # Lists of tensors size (batch_size, *).
            x = data["x"]
            y = data["y"]
            list_dataset = ListDataset(x, y)
            loader = DataLoader(list_dataset,
                                batch_size=None,
                                shuffle=True)

        # Dict for logging optimisation progress.
        training_curve = defaultdict(list)

        # Dict for logging performance progress.
        performance_curve = defaultdict(list)

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epochs")
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                self.communications += 1
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(p).sum() / len(self.data["x"])

                # Compute E_q[log p(y | x, θ)].
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]).sum()
                ll /= len(x_batch)

                # Negative local Free Energy is KL minus log-probability
                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"],
                                   lr=optimiser.param_groups[0]["lr"])

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % self.config["print_epochs"] == 0 \
                    or i == (self.config["epochs"] - 1):
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

                # Update global posterior before making predictions.
                self.q = q.non_trainable_copy()

                # Get training set performance.
                report = f"Epochs: {i}. ELBO: {epoch['elbo']:.3f} "\
                         f"LL: {epoch['ll']:.3f} KL: {epoch['kl']:.3f} "

                train_pp = self.model_predict(self.data["x"])
                train_mll = train_pp.log_prob(self.data["y"]).mean()
                report += f"Train mll: {train_mll:.3f} "
                performance_curve["epoch"].append(i)
                performance_curve["train_mll"].append(train_mll.item())

                # Get validation set performance.
                if self.val_data is not None:
                    val_pp = self.model_predict(self.val_data["x"])
                    val_mll = val_pp.log_prob(self.val_data["y"]).mean()
                    report += f"Val mll: {val_mll:.3f} "
                    performance_curve["val_mll"].append(val_mll.item())

                tqdm.write(report)

            # Log q so we can track performance each epoch.
            self.log["q"].append(q.non_trainable_copy())
            self.log["communications"].append(self.communications)
            self.epochs += 1

            # Update learning rate.
            lr_scheduler.step()

        # Update global posterior.
        self.q = q.non_trainable_copy()

        self.iterations += 1

        # Log training.
        self.log["training_curves"].append(training_curve)

        # Log performance.
        self.log["performance_curves"].append(performance_curve)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
