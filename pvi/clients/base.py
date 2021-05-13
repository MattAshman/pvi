import logging
import torch

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Client class
# =============================================================================


class Client:
    def __init__(self, data, model, t=None, config=None):

        if config is None:
            config = {}

        self._config = self.get_default_config()
        self.config = config
        
        # Set data partition and likelihood
        self.data = data
        self.model = model
        
        # Set likelihood approximating term
        self.t = t
        
        self.log = defaultdict(list)
        self._can_update = True

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    @classmethod
    def get_default_config(cls):
        return {
            "train_model": False,
            "damping_factor": 1.,
            "valid_factors": False,
            "update_log_coeff": False,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "lr_scheduler": "MultiplicativeLR",
            "lr_scheduler_params": {
                "lr_lambda": lambda epoch: 1.
            },
            "num_elbo_samples": 10,
            "print_epochs": 1
        }

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        """
        return self._can_update
    
    def fit(self, q, init_q=None):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(q, init_q)

    def update_q(self, q, init_q=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        # Type(q) is self.model.conjugate_family.
        if str(type(q)) == str(self.model.conjugate_family) \
                and not self.config["train_model"]:
            # No need to make q trainable.
            q_new, self.t = self.model.conjugate_update(self.data, q, self.t)
        else:
            # Pass a trainable copy to optimise.
            q_new, self.t = self.gradient_based_update(p=q, init_q=init_q)

        return q_new, self.t

    def gradient_based_update(self, p, init_q=None):
        # Cannot update during optimisation.
        self._can_update = False
        
        # Copy the approximate posterior, make non-trainable.
        q_old = p.non_trainable_copy()
        q_cav = p.non_trainable_copy()

        if self.t is not None:
            # TODO: check if valid distribution.
            q_cav.nat_params = {k: v - self.t.nat_params[k]
                                for k, v in q_cav.nat_params.items()}

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters(),
                     **self.config["model_optimiser_params"]}
                ]
            else:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters()}
                ]
        else:
            parameters = q.parameters()

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

        # Dict for logging optimisation progress
        training_curve = defaultdict(list)
        
        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                          leave=True)
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo" : 0,
                "kl"   : 0,
                "ll"   : 0,
                "logt" : 0,
            }
            
            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x" : x_batch,
                    "y" : y_batch,
                }

                # TODO: Put this code inside a unit test.
                # Compute KL divergence between q and q_old.
                # kl1 = q.kl_divergence(q_old).sum() / len(x)

                # Compute KL divergence between q and q_old using
                # torch.distribution.
                # kl2 = torch.distributions.kl_divergence(
                #     q.distribution, q_old.distribution).sum() / len(x)

                # Compute KL divergence between q and q_cav.
                # kl3 = q.kl_divergence(q_cav).sum() / len(x)

                # Compute KL divergence between q and q_cav using
                # torch.distribution
                # kl4 = torch.distributions.kl_divergence(
                #     q.distribution, q_cav.distribution).sum() / len(x)

                # Compute the KL divergence between q and q_cav, ignoring
                # A(η_cav).
                kl5 = q.kl_divergence(q_cav, calc_log_ap=False).sum() / len(x)

                # Sample θ from q and compute p(y | θ, x) for each θ
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]).sum()
                ll /= len(x_batch)

                # Compute E_q[log t(θ)].
                # logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                # logt /= len(x)

                # Negative local free energy is KL minus log-probability.
                # loss1 = kl1 + logt - ll
                # loss2 = kl2 + logt - ll
                # loss3 = kl3 - ll
                # loss4 = kl4 - ll
                loss5 = kl5 - ll
                # losses = [loss1, loss2, loss3, loss4, loss5]

                # Compute gradients for all 5 losses.
                # grads = []
                # for loss in losses:
                #     loss.backward(retain_graph=True)
                #     grads.append({k: v.grad.clone()
                #                   for k, v in q._unc_params.items()})
                #     optimiser.zero_grad()

                # Check all gradients are equal.
                # for p in q._unc_params.keys():
                #     for j in range(len(grads)):
                #         for k in range(len(grads)):
                #             if not torch.allclose(grads[j][p], grads[k][p]):
                #                 import pdb
                #                 pdb.set_trace()
                #                 raise ValueError("Gradients not equal!")

                # Use loss 5.
                loss5.backward()
                optimiser.step()
                
                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss5.item() / len(loader)
                epoch["kl"] += kl5.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)
                # epoch["logt"] += logt.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if self.t is not None:
                training_curve["logt"].append(epoch["logt"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             # f"log t: {epoch['logt']:.3f}, "
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"], # logt=epoch["logt"],
                                   lr=optimiser.param_groups[0]["lr"])

            # Update learning rate.
            lr_scheduler.step()

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Create non-trainable copy to send back to server.
        q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q, q_old, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"])

            return q_new, t_new

        else:
            return q_new, None


class ClientBayesianHypers:
    """
    PVI client with Bayesian treatment of model hyperparameters.
    """
    def __init__(self, data, model, t=None, teps=None, config=None):

        if config is None:
            config = {}

        self._config = self.get_default_config()
        self.config = config

        # Set data partition and likelihood
        self.data = data
        self.model = model

        # Set likelihood approximating term
        self.t = t
        self.teps = teps

        self.log = defaultdict(list)
        self._can_update = True

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    @classmethod
    def get_default_config(cls):
        return {
            "damping_factor": 1.,
            "valid_factors": False,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "num_elbo_samples": 10,
            "num_elbo_hyper_samples": 1,
            "print_epochs": 1
        }

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        :return:
        """
        return self._can_update

    def fit(self, q, qeps):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(q, qeps)

    def update_q(self, q, qeps):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """
        # Pass a trainable copy to optimise.
        q_new, qeps_new, self.t, self.teps = self.gradient_based_update(
            q.trainable_copy(), qeps.trainable_copy())

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
            q_cav.nat_params = {k: v - self.t.nat_params[k]
                                for k, v in q_cav.nat_params.items()}

            qeps_cav = qeps.non_trainable_copy()
            for k1, v1 in qeps.distributions.items():
                qeps.distributions[k1].nat_params = {
                    k2: v2 - self.teps.factors[k1].nat_params[k2]
                    for k2, v2 in v1.nat_params.items()}

        parameters = list(q.parameters()) + qeps.parameters()

        # Reset optimiser. Parameters are those of q(θ) and q(ε).
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

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
                        batch, q, self.config["num_elbo_samples"]).sum()

                ll /= (self.config["num_elbo_hyper_samples"] * len(x_batch))

                if self.t is not None:
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    logteps = sum(self.teps.eqlogt(
                        qeps, self.config["num_elbo_samples"]).values())
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

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"KL eps: {epoch['kleps']:.3f}, "
                             f"log t: {epoch['logt']: .3f},"
                             f"log teps: {epoch['logteps']: .3f},"
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"], kleps=epoch["kleps"],
                                   logt=epoch["logt"],
                                   logteps=epoch["logteps"])

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
                q, q_cav, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"])
            teps_new = self.teps.compute_refined_factor(
                qeps, qeps_cav, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"])

            return q_new, qeps_new, t_new, teps_new

        else:
            return q_new, qeps_new, None, None