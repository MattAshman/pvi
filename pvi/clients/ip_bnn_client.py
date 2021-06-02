import logging
import torch

from .base import Client
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class IPBNNClient(Client):

    def gradient_based_update(self, p, init_q=None):
        # TODO: Can this somehow be merged with the base Client class?
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make non-trainable.
        if init_q is None:
            q_cav = p.non_trainable_copy()
            q = p.non_trainable_copy()
        else:
            q_cav = init_q.non_trainable_copy()
            q = init_q.non_trainable_copy()

        q_cav, t_idx = q_cav.form_cavity(self.t)

        # Replace old factor with new (optimisable) factor.
        t_new = self.t.trainable_copy()
        q.ts[t_idx] = t_new

        # Parameters are those of t_new(θ) and self.model.
        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": t_new.parameters()},
                    {"params": self.model.parameters(),
                     **self.config["model_optimiser_params"]}
                ]
            else:
                parameters = [
                    {"params": t_new.parameters()},
                    {"params": self.model.parameters()}
                ]
        else:
            parameters = t_new.parameters()

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                          leave=True)
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.)

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                ll, kl = self.model.elbo(
                    batch, q_cav, q,
                    num_samples=self.config["num_elbo_samples"])
                kl /= len(x)
                ll /= len(x_batch)

                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"])

            # Log progress for current epoch
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["kl"].append(epoch["kl"])
            training_metrics["ll"].append(epoch["ll"])

            if i > 0 and i % self.config["print_epochs"] == 0:
                # Update global posterior before evaluating performance.
                self.q = q.non_trainable_copy()

                metrics = self.evaluate_performance({
                    "epochs": i,
                    "elbo": epoch["elbo"],
                    "kl": epoch["kl"],
                    "ll": epoch["ll"],
                })

                # Report performance.
                report = ""
                for k, v in metrics.items():
                    report += f"{k}: {v:.3f} "
                    performance_metrics[k].append(v)

                tqdm.write(report)

            # Update learning rate.
            lr_scheduler.step()

            # Check whether to stop early.
            if self.config["early_stopping"](training_metrics["elbo"]):
                break

        # Log the training curves for this update
        self.log["training_curves"].append(training_metrics)
        self.log["performance_curves"].append(performance_metrics)

        # Create non-trainable copy to send back to server.
        q_new = q.non_trainable_copy()
        t_new = t_new.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        return q_new, t_new
