import logging
import torch

from collections import defaultdict
from .base import Client
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.utils.gaussian import joint_from_marginal, nat_from_std, std_from_nat
from pvi.utils.gaussian_extra import joint_from_marginal_lingauss
from pvi.distributions import MultivariateGaussianDistributionWithZ
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import pdb

logger = logging.getLogger(__name__)

JITTER = 1e-6


class FederatedSGPClientNoProjection(Client):
    def __init__(self, data, model, t, config=None):
        super().__init__(data, model, t, config)

    def gradient_based_update(self, p, init_q=None):
        # TODO: use the collapsed variational lower bound for SGP regression.
        #  i.e. should be closed-form solution.
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make old posterior non-trainable.
        q_cav = p.non_trainable_copy()

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # q = p.trainable_copy()
            z = self.t.inducing_locations
            M = z.shape[0]
            q = MultivariateGaussianDistributionWithZ(
                nat_params={
                    "np1": torch.zeros(M).double(),
                    "np2": -0.001 * torch.ones(M).double().diag_embed(),
                },
                inducing_locations=z,
                train_inducing=True,
                is_trainable=True,
            )

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": q.parameters()},
                    {
                        "params": self.model.parameters(),
                        **self.config["model_optimiser_params"],
                    },
                ]
            else:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters()},
                ]
        else:
            parameters = q.parameters()

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )
        lr_scheduler = getattr(
            torch.optim.lr_scheduler, self.config["lr_scheduler"]
        )(optimiser, **self.config["lr_scheduler_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(
            tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Dict for logging performance progress.
        performance_metrics = defaultdict(list)

        # Stay fixed throughout optimisation as model hyperparameters are
        # only updated by the server.
        za = q_cav.inducing_locations
        kaa = add_diagonal(self.model.kernel(za, za).detach(), JITTER)
        ikaa = psd_inverse(kaa)

        # Compute joint distributions q(a, b) and qcav(a, b).
        qab, qab_cav = self.compute_q(q, za, q_cav, ikaa)

        # Gradient-based optimisation loop.
        epoch_iter = tqdm(
            range(self.config["epochs"]), desc="Epoch", leave=True
        )
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and q_old.
                # kl = qab.kl_divergence(qab_old).sum() / len(x)
                kl = qab.kl_divergence(qab_cav, calc_log_ap=False).sum()
                kl /= len(x)

                # Sample θ from q and compute p(y | θ, x) for each θ.
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]
                ).sum()
                ll /= len(x_batch)

                loss = kl - ll
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                # Recompute joint distributions q(a, b) and qcav(a, b).
                qab, qab_cav = self.compute_q(q, za, q_cav, ikaa)

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(
                elbo=epoch["elbo"],
                kl=epoch["kl"],
                ll=epoch["ll"],
                logt=epoch["logt"],
            )

            # Log progress for current epoch
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["kl"].append(epoch["kl"])
            training_metrics["ll"].append(epoch["ll"])

            if i > 0 and i % self.config["print_epochs"] == 0:
                # Update global posterior before evaluating performance.
                self.q = q.non_trainable_copy()

                metrics = self.evaluate_performance(
                    {
                        "epochs": i,
                        "elbo": epoch["elbo"],
                        "kl": epoch["kl"],
                        "ll": epoch["ll"],
                    }
                )

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

        # Log the training curves for this update.
        self.log["training_curves"].append(training_metrics)
        self.log["performance_curves"].append(performance_metrics)

        # Create non-trainable copy to send back to server.
        q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        # Compute new local contribution
        zb = q_new.inducing_locations
        kab = self.model.kernel(za, zb)
        kbb = add_diagonal(self.model.kernel(zb, zb), JITTER)
        a = kab.transpose(-1, -2).matmul(ikaa)
        c = kbb - a.matmul(kab)
        ic = psd_inverse(c)
        prec = q_new.std_params["covariance_matrix"].inverse()
        m = q_new.std_params["loc"]
        tb_np2 = -0.5 * (prec - ic)
        tb_np1 = prec.matmul(m)
        tb_np = {"np1": tb_np1, "np2": tb_np2}

        t_new = type(self.t)(
            inducing_locations=zb.detach(),
            nat_params=tb_np,
        )

        return q_new, t_new

    def compute_q(self, q, za, q_cav, ikaa):
        # Zb are new private inducing locations, which are to be optimised.
        zb = q.inducing_locations
        z = torch.cat([za, zb], axis=0)

        kab = self.model.kernel(za, zb)
        kbb = add_diagonal(self.model.kernel(zb, zb), JITTER)

        # q_cav(a, b) = q_cav(a) p(b | a).
        qab_cav_loc, qab_cav_cov = joint_from_marginal(
            q_cav, kab, kbb=kbb, ikaa=ikaa
        )

        qab_cav_np = nat_from_std(
            std_params={
                "loc": qab_cav_loc,
                "covariance_matrix": qab_cav_cov,
            }
        )
        qab_cav = type(q)(inducing_locations=z, nat_params=qab_cav_np)
        pdb.set_trace()

        # q(a, b) = q_cav(a) q(b | a).
        qab_loc, qab_cov = joint_from_marginal_lingauss(
            q_cav, q, kab, kbb=kbb, ikaa=ikaa
        )

        qab = type(q)(
            inducing_locations=z,
            std_params={
                "loc": qab_loc,
                "covariance_matrix": qab_cov,
            },
        )
        pdb.set_trace()
        return qab, qab_cav
