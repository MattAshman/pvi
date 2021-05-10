import torch

from torch import nn


class BNNDistribution:
    """
    Maintains a distribution over each layer of a BNN.
    """
    def __init__(self, distributions):

        self.distributions = distributions

    def non_trainable_copy(self):
        distributions = [dist.non_trainable_copy()
                         for dist in self.distributions]

        return type(self)(distributions)

    def trainable_copy(self):
        distributions = [dist.trainable_copy()
                         for dist in self.distributions]

        return type(self)(distributions)

    def compute_dist(self, layer, act_z):
        return self.distributions[layer]


class BNNFactor:
    """
    Maintains a pseudo-likelihood factor over each layer of a BNN.
    """
    def __init__(self, distributions, inducing_locations, train_inducing=True):

        self.distributions = distributions
        self.train_inducing = train_inducing

        if inducing_locations is not None:
            self._inducing_locations = nn.Parameter(
                inducing_locations, requires_grad=self.train_inducing)
        else:
            self._inducing_locations = inducing_locations

    @property
    def inducing_locations(self):
        return self._inducing_locations

    @inducing_locations.setter
    def inducing_locations(self, value):
        self._inducing_locations = nn.Parameter(
            value, requires_grad=self.train_inducing)

    def non_trainable_copy(self):
        distributions = [dist.non_trainable_copy()
                         for dist in self.distributions]

        if self._inducing_locations is not None:
            inducing_locations = self.inducing_locations.detach().clone()
        else:
            inducing_locations = None

        return type(self)(
            distributions, inducing_locations,
            train_inducing=self.train_inducing
        )

    def trainable_copy(self):
        distributions = [dist.trainable_copy()
                         for dist in self.distributions]

        if self._inducing_locations is not None:
            inducing_locations = self.inducing_locations.detach().clone()
        else:
            inducing_locations = None

        return type(self)(
            distributions, inducing_locations,
            train_inducing=self.train_inducing
        )

    def parameters(self):
        parameters = [list(dist.parameters()) for dist in self.distributions]
        return [item for sublist in parameters for item in sublist]


class IPBNNGaussianPosterior:
    """
    Maintains the distribution q({w_l}) = p({w_l}) Π t({w_l}).
    """
    def __init__(self, p, ts):
        self.p = p
        self.ts = ts

    @property
    def inducing_locations(self):
        inducing_locations = torch.cat([t.inducing_locations for t in self.ts])
        return inducing_locations

    def compute_cavity(self, t):
        """
        Returns the distribution q({w_l}) = p({w_l}) Π _{/ i} t({w_l}).
        :param t: Pseudo-likelihood factor to remove from self.ts.
        :return: q({w_l}) = p({w_l}) Π _{/ i} t({w_l}).
        """
        # Find the pseudo-likelihood factor in self.ts and remove.
        ts = self.ts
        for i, ti in self.ts:
            same_inducing = torch.allclose(
                ti.inducing_locations, t.inducing_locations)
            same_np1 = torch.allclose(
                ti.nat_params["np1"], t.nat_params["np1"])
            same_np2 = torch.allclose(
                ti.nat_params["np2"], t.nat_params["np2"])

            if same_inducing and same_np1 and same_np2:
                ts.pop(i)
                break

        return type(self)(p=self.p, ts=ts)

    def compute_dist(self, layer, act_z):
        """
        Compute the distribution q(w_l | {w_l}) =
        :param layer: Layer for which to compute the distribution at.
        :param act_z: Post-activation Φ(z), (m, dim_in).
        :return: q(w_l), (dim_out).
        """
        # TODO: this assumes both prior and factors are mean-field.

        # Get IP means and variances for layer. Each t_dist maintains a
        # distribution with dimension (mi, dim_out).
        t_dists = [t.distributions[layer] for t in self.ts]
        p_dist = self.p.distributions[layer]

        # Assumes t distributions are mean-field Gaussians.
        # (dim_out, m).
        t_np1 = torch.cat([dist.nat_params["np1"] for dist in t_dists], dim=0)
        t_np2 = torch.cat([dist.nat_params["np2"] for dist in t_dists], dim=0)
        # (dim_out, m).
        t_np1 = t_np1.transpose(0, 1)
        t_np2 = t_np2.transpose(0, 1)
        # (dim_out, m, m).
        t_np2 = t_np2.diag_embed()

        # (dim_in, m) x (dim_out, m, 1) -> (dim_out, dim_in, 1).
        np1 = act_z.T.matmul(t_np1.unsqueeze(-1))

        # (dim_in, m) x (dim_out, m, m) x (m, dim_in)
        # -> (dim_out, dim_in, dim_in).
        np2 = (p_dist.nat_params["np2"].diag_embed()
               + act_z.T.matmul(t_np2).matmul(act_z))

        # (dim_out, dim_in) dimensional distributions.
        cov = (-0.5 * np2).inverse()
        loc = -0.5 * cov.matmul(np1).squeeze()
        qw = torch.distributions.MultivariateNormal(loc, cov)

        return qw

    def non_trainable_copy(self):
        return type(self)(
            p=self.p.non_trainable_copy(),
            ts=[t.non_trainable_copy() for t in self.ts],
        )

    def trainable_copy(self):
        # TODO: Never train prior distribution??
        return type(self)(
            p=self.p.non_trainable_copy(),
            ts=[t.trainable_copy() for t in self.ts],
        )

    def parameters(self):
        parameters = [t.parameters() for t in self.ts]
        return [item for sublist in parameters for item in sublist]
