import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from pvi.utils.training_utils import EarlyStopping

# =============================================================================
# Client class
# =============================================================================


class Client:
    def __init__(self, data, model, t=None, config=None, val_data=None):

        if config is None:
            config = {}

        self._config = self.get_default_config()
        self.config = config

        # Set data partition and likelihood
        self.data = data
        self.model = model

        # Set likelihood approximating term
        self.t = t

        # Maintain optimised approximate posterior.
        self.q = None

        # Validation dataset.
        self.val_data = val_data

        self.log = defaultdict(list)
        self._can_update = True

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    def get_default_config(self):
        return {
            "train_model": False,
            "damping_factor": 1.0,
            "valid_factors": False,
            "update_log_coeff": False,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "model_optimiser_params": {},
            "early_stopping": EarlyStopping(np.inf),
            "performance_metrics": None,
            "num_elbo_samples": 10,
            "print_epochs": np.inf,
            "device": "cpu",
            "verbose": False,
            "log_training": True,
            "log_performance": True,
        }

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        """
        return self._can_update

    def evaluate_performance(self, default_metrics=None):
        metrics = {}
        if default_metrics is not None:
            metrics = {**default_metrics, **metrics}

        if self.config["performance_metrics"] is not None:
            train_metrics = self.config["performance_metrics"](self, self.data)
            for k, v in train_metrics.items():
                metrics["train_" + k] = v

            if self.val_data is not None:
                val_metrics = self.config["performance_metrics"](self, self.val_data)
                for k, v in val_metrics.items():
                    metrics["val_" + k] = v

        return metrics

    def fit(self, *args, **kwargs):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(*args, **kwargs)

    def update_q(self, q, init_q=None, **kwargs):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """
        # Type(q) is self.model.conjugate_family.
        if (
            str(type(q)) == str(self.model.conjugate_family)
            and not self.config["train_model"]
        ):
            # No need to make q trainable.
            self.q, self.t = self.model.conjugate_update(self.data, q, self.t)
        else:
            # Pass a trainable copy to optimise.
            self.q, self.t = self.gradient_based_update(p=q, init_q=init_q, **kwargs)

        return self.q, self.t

    def gradient_based_update(self, p, init_q=None, **kwargs):
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make non-trainable.
        q_old = p.non_trainable_copy()
        # Cavity distribution.
        q_cav = p.non_trainable_copy()

        if self.t is not None:
            # TODO: check if valid distribution.
            q_cav.nat_params = {
                k: v - self.t.nat_params[k] for k, v in q_cav.nat_params.items()
            }

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            parameters = [
                {"params": list(q.parameters())},
                {
                    "params": self.model.parameters(),
                    **self.config["model_optimiser_params"],
                },
            ]
        else:
            parameters = list(q.parameters())

        # Reset optimiser.
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )

        # Set up data loader.
        tensor_dataset = TensorDataset(*self.data.values())
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        # Dicts for logging optimisation progress.
        training_metrics = defaultdict(list)
        performance_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](scores=None, model=q.non_trainable_copy())

        # Gradient-based optimisation loop -- loop over epochs.
        epoch_iter = tqdm(
            range(self.config["epochs"]),
            desc="Epoch",
            leave=True,
            disable=(not self.config["verbose"]),
        )
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)
            for batch in iter(loader):
                batch = {
                    k: batch[i].to(self.config["device"])
                    for i, k in enumerate(self.data.keys())
                }

                optimiser.zero_grad()

                # Compute the KL divergence between q and q_cav, ignoring A(η_cav).
                kl = q.kl_divergence(q_cav, calc_log_ap=False).sum() / len(
                    self.data["x"]
                )

                # Compute expected log likelihood under q.
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
                (i > 0 and i % self.config["print_epochs"] == 0)
                or i == (self.config["epochs"] - 1)
                or stop_early
            ):
                # Update global posterior before evaluating performance.
                self.q = q.non_trainable_copy()

                metrics = self.evaluate_performance({"epochs": i, **epoch})
                for k, v in metrics.items():
                    performance_metrics[k].append(v)

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

            # Check whether to stop early.
            if stop_early:
                break

        # Log the training curves for this update.
        if self.config["log_training"]:
            self.log["training_curves"].append(training_metrics)
        if self.config["log_performance"]:
            self.log["performance_curves"].append(performance_metrics)

        # Create non-trainable copy to send back to server.
        if self.config["early_stopping"].stash_model:
            q_new = self.config["early_stopping"].best_model
        else:
            q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions.
            t_new = self.t.compute_refined_factor(
                q_new,
                q_old,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"],
            )

            return q_new, t_new

        else:
            return q_new, None

    def model_predict(self, x, **kwargs):
        """
        Returns the current models predictive posterior distribution.
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        return self.model(x, self.q, **kwargs)


class ClientBayesianHypers(Client):
    """
    PVI client with Bayesian treatment of model hyperparameters.
    """

    def __init__(self, teps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set likelihood approximating term
        self.teps = teps

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "num_elbo_hyper_samples": 1,
        }

    def update_q(self, q, qeps, init_q=None, init_qeps=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """
        # Pass a trainable copy to optimise.
        q_new, qeps_new, self.t, self.teps = self.gradient_based_update(
            q.trainable_copy(), qeps.trainable_copy()
        )

        return q_new, qeps_new, self.t, self.teps

    def gradient_based_update(self, q, qeps):
        # Cannot update during optimisation.
        self._can_update = False

        if self.t is None:
            # Old posterior = prior, make non-trainable.
            q_cav = q.non_trainable_copy()
            qeps_cav = qeps.non_trainable_copy()
        else:
            q_cav = q.non_trainable_copy()
            q_cav.nat_params = {
                k: v - self.t.nat_params[k] for k, v in q_cav.nat_params.items()
            }

            qeps_cav = qeps.non_trainable_copy()
            for k1, v1 in qeps.distributions.items():
                qeps.distributions[k1].nat_params = {
                    k2: v2 - self.teps.factors[k1].nat_params[k2]
                    for k2, v2 in v1.nat_params.items()
                }

        parameters = list(q.parameters()) + qeps.parameters()

        # Reset optimiser. Parameters are those of q(θ) and q(ε).
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "kleps": [],
            "ll": [],
            "logt": [],
            "logteps": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch")
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "kleps": 0,
                "ll": 0,
                "logt": 0,
                "logteps": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(q_cav).sum() / len(x)
                kleps = sum(qeps.kl_divergence(qeps_cav).values()) / len(x)

                # Estimate E_q[log p(y | x, θ, ε)].
                ll = 0
                for _ in range(self.config["num_elbo_hyper_samples"]):
                    eps = qeps.rsample()
                    self.model.hyperparameters = eps
                    ll += self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]
                    ).sum()

                ll /= self.config["num_elbo_hyper_samples"] * len(x_batch)

                if self.t is not None:
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    logteps = sum(
                        self.teps.eqlogt(qeps, self.config["num_elbo_samples"]).values()
                    )
                    logt /= len(x)
                    logteps /= len(x)

                    # loss = kl + kleps - ll + logt - logteps

                loss = kl + kleps - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["kleps"] += kleps.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

                if self.t is not None:
                    epoch["logt"] += logt.item() / len(loader)
                    epoch["logteps"] += logteps.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["kleps"].append(epoch["kleps"])
            training_curve["ll"].append(epoch["ll"])

            if self.t is not None:
                training_curve["logt"].append(epoch["logt"])
                training_curve["logteps"].append(epoch["logteps"])

            epoch_iter.set_postfix(
                elbo=epoch["elbo"],
                kl=epoch["kl"],
                ll=epoch["ll"],
                kleps=epoch["kleps"],
                logt=epoch["logt"],
                logteps=epoch["logteps"],
            )

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Create non_trainable_copy to send back to server.
        q_new = q.non_trainable_copy()
        qeps_new = qeps.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q,
                q_cav,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
            )
            teps_new = self.teps.compute_refined_factor(
                qeps,
                qeps_cav,
                damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
            )

            return q_new, qeps_new, t_new, teps_new

        else:
            return q_new, qeps_new, None, None

    def model_predict(self, x, **kwargs):
        raise NotImplementedError
