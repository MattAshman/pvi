import torch

from torch import nn
from pvi.utils.psd_utils import psd_inverse, add_diagonal


class BNNDistribution(nn.Module):
    """
    Maintains a distribution over each layer of a BNN.
    """
    def __init__(self, distributions):
        super().__init__()
        self.distributions = nn.ModuleList(distributions)

    def non_trainable_copy(self):
        distributions = [dist.non_trainable_copy()
                         for dist in self.distributions]

        return type(self)(distributions)

    def trainable_copy(self):
        distributions = [dist.trainable_copy()
                         for dist in self.distributions]

        return type(self)(distributions)

    def compute_dist(self, layer, *args, **kwargs):
        return self.distributions[layer]

    def log_prob(self, layer, theta, *args, **kwargs):
        return self.distributions[layer].log_prob(theta.transpose(-1, -2))


class BNNFactor(nn.Module):
    """
    Maintains a pseudo-likelihood factor over each layer of a BNN.
    """
    def __init__(self, distributions, inducing_locations, is_trainable=False,
                 train_inducing=True):
        super().__init__()

        self.distributions = nn.ModuleList(distributions)
        self.train_inducing = train_inducing

        if is_trainable and inducing_locations is not None:
            self._inducing_locations = nn.Parameter(
                inducing_locations, requires_grad=self.train_inducing)
        else:
            self._inducing_locations = inducing_locations

    @property
    def inducing_locations(self):
        return self._inducing_locations

    @inducing_locations.setter
    def inducing_locations(self, value):
        if self.is_trainable:
            self._inducing_locations = nn.Parameter(
                value, requires_grad=self.train_inducing)
        else:
            self._inducing_locations = value

    def non_trainable_copy(self):
        distributions = [dist.non_trainable_copy()
                         for dist in self.distributions]

        if self._inducing_locations is not None:
            inducing_locations = self.inducing_locations.detach().clone()
        else:
            inducing_locations = None

        return type(self)(
            distributions, inducing_locations, is_trainable=False,
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
            distributions, inducing_locations, is_trainable=True,
            train_inducing=self.train_inducing
        )


class IPBNNGaussianPosterior(nn.Module):
    """
    Maintains the distribution q({w_l}) = p({w_l}) Π t({w_l}).
    """
    def __init__(self, p, ts):
        super().__init__()

        self.p = p
        self.ts = nn.ModuleList(ts)

    @property
    def inducing_locations(self):
        inducing_locations = torch.cat([t.inducing_locations for t in self.ts])
        return inducing_locations

    def form_cavity(self, t):
        """
        Returns the distribution q({w_l}) = p({w_l}) Π _{/ i} t({w_l}).
        :param t: Pseudo-likelihood factor to remove from self.ts.
        :return: q({w_l}) = p({w_l}) Π _{/ i} t({w_l}).
        """
        # Find the pseudo-likelihood factor in self.ts and remove.
        ts = self.ts
        for i, ti in enumerate(self.ts):
            same_inducing = torch.allclose(
                ti.inducing_locations, t.inducing_locations)

            same_np1, same_np2 = [], []
            for ti_dist, t_dist in zip(ti.distributions, t.distributions):
                same_np1.append(torch.allclose(
                    ti_dist.nat_params["np1"], t_dist.nat_params["np1"]))
                same_np2.append(torch.allclose(
                    ti_dist.nat_params["np2"], t_dist.nat_params["np2"]))

            if same_inducing and all(same_np1) and all(same_np2):
                # Set natural parameters to 0. We retain it as need to keep
                # inducing point values.
                for dist in ts[i].distributions:
                    for k, v in dist.nat_params.items():
                        dist.nat_params[k] = torch.zeros_like(v)

                return type(self)(p=self.p, ts=ts), i

        raise ValueError("Could not find t in self.ts!")

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

        # (m, dim_out).
        t_np1 = torch.cat([dist.nat_params["np1"] for dist in t_dists], dim=0)
        t_np2 = torch.cat([dist.nat_params["np2"] for dist in t_dists], dim=0)

        # (dim_out, m).
        t_np1 = t_np1.transpose(0, 1)
        t_np2 = t_np2.transpose(0, 1)

        # (dim_out, m, m).
        t_np2 = t_np2.diag_embed()

        # (num_samples, dim_out, m, dim_in).
        act_z_ = act_z.unsqueeze(1)
        # (num_samples, dim_out, m).
        t_np1_ = t_np1.unsqueeze(0)
        # (num_samples, dim_out, dim_in, 1)
        np1 = act_z_.transpose(-1, -2).matmul(t_np1_.unsqueeze(-1))

        # (num_samples, dim_out, m, m).
        t_np2_ = t_np2.unsqueeze(0)
        # (m, m).
        p_np2 = p_dist.nat_params["np2"].diag_embed()
        # (num_samples, dim_out, dim_in, dim_in).
        np2 = p_np2 + act_z_.transpose(-1, -2).matmul(t_np2_).matmul(act_z_)

        # Compute mean and covariance matrix for each column of weights.
        # TODO: currently performs two cholesky inversions when only one is
        #  needed.
        prec = -2. * np2
        cov = psd_inverse(prec)
        loc = cov.matmul(np1).squeeze(-1)
        qw = torch.distributions.MultivariateNormal(loc, cov)

        return qw

    def log_prob(self, layer, act_z, theta, *args, **kwargs):
        return self.compute_dist(layer, act_z).log_prob(
            theta.transpose(-1, -2)).sum(-1)

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

    def replace_factor(self, t_old, t_new, **kwargs):
        """
        Forms a new distribution by replacing t_old(θ) with t_new(θ).
        :param t_old: The factor to remove.
        :param t_new: The factor to add.
        :param kwargs: Passed to self.create_new()
        :return: Updated distribution.
        """
        # Find location of old factor and replace.
        q, t_idx = self.form_cavity(t_old)
        q.ts[t_idx] = t_new

        return q


class PseudoObservationNormalPrior(nn.Module):
    """
    Implements the 'pseudo-observation' prior distribution over weights,
    p(W | Z) = Π_d N(w_d; 0, σ^2 I) N(Zw_d; 0, τ^2 * I).
    """
    def __init__(self, prior_scale, obs_scale):
        super().__init__()

        self.prior_scale = prior_scale
        self.obs_scale = obs_scale

    def compute_dist(self, layer, act_z):
        """
        Computes p(W | Z) = Π_d N(w_d; 0, σ^2 I) N(w_d; 0, τ^2 (Z^T * Z)^{-1}).
        :param layer: Layer for which to compute the distribution at.
        :param act_z: Post-activation Φ(z), (m, dim_in).
        :return: p(W | Z) = Π_d N(w_d; 0, σ^2 I) N(w_d; 0, σ^2 Z^T * Z)
        """
        prec = self.obs_scale.pow(-2) * act_z.transpose(-1, -2).matmul(act_z)
        prec += self.prior_scale.pow(-2) * torch.ones(
            *prec.shape[:-1]).diag_embed()
        cov = psd_inverse(prec)
        loc = torch.zeros(*cov.shape[:-1])
        pw = torch.distributions.MultivariateNormal(loc, cov)

        return pw

    def log_prob(self, layer, act_z, theta, *args, **kwargs):
        return self.compute_dist(layer, act_z).log_prob(
            theta.transpose(-1, -2)).sum(-1)

    def non_trainable_copy(self):
        return self

    def trainable_copy(self):
        return self


class BatchNormNormalPrior(nn.Module):
    """
    Implements the 'batch-norm' prior distribution over weights,
    p(W | Z) = Π_d N(Zw_d; 0, σ^2 * I).
    """
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def compute_dist(self, layer, act_z):
        """
        Computes p(W | Z) = Π_d N(w_d; 0, σ^2 (Z^T * Z)^{-1}).
        :param layer: Layer for which to compute the distribution at.
        :param act_z: Post-activation Φ(z), (m, dim_in).
        :return: p(W | Z) = Π_d N(w_d; 0, σ^2 Z^T * Z)
        """
        prec = self.scale.pow(-2) * act_z.transpose(-1, -2).matmul(act_z)
        cov = psd_inverse(prec)
        loc = torch.zeros(*cov.shape[:-1])
        pw = torch.distributions.MultivariateNormal(loc, cov)

        return pw

    def log_prob(self, layer, act_z, theta, *args, **kwargs):
        # Theta is (num_samples, dim_out, dim_in).
        # (num_samples, m, dim_in) x (num_samples, dim_in, dim_out)
        # -> (num_samples, m, dim_out) -> (num_samples, dim_out, m).
        # Note: this is more efficient than computing the Gaussian over p(w).
        import pdb
        pdb.set_trace()
        zw = act_z.matmul(theta)
        pzw = torch.distributions.Normal(
            torch.zeros(*zw.shape[1:]).to(zw),
            torch.ones(*zw.shape[1:]).to(zw) * self.scale
        )
        return pzw.log_prob(zw)

    def non_trainable_copy(self):
        return self

    def trainable_copy(self):
        return self


class IPBNNBatchNormGaussianPosterior(IPBNNGaussianPosterior):
    """
    Implements the 'batch-norm' Gaussian posterior distribution over weights.
    """

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

        # (m, dim_out).
        t_np1 = torch.cat([dist.nat_params["np1"] for dist in t_dists], dim=0)
        t_np2 = torch.cat([dist.nat_params["np2"] for dist in t_dists], dim=0)

        # (dim_out, m).
        t_np1 = t_np1.transpose(0, 1)
        t_np2 = t_np2.transpose(0, 1)

        # (dim_out, m, m).
        t_np2 = t_np2.diag_embed()

        # (num_samples, 1, m, dim_in).
        act_z_ = act_z.unsqueeze(1)
        # (1, dim_out, m).
        t_np1_ = t_np1.unsqueeze(0)
        # (num_samples, dim_out, dim_in, 1)
        np1 = act_z_.transpose(-1, -2).matmul(t_np1_.unsqueeze(-1))

        # (1, dim_out, m, m).
        t_np2_ = t_np2.unsqueeze(0)

        # (num_samples, 1, dim_in, dim_in).
        # p_np2 = act_z.transpose(-1, -2).matmul(act_z).unsqueeze(1)
        # p_np2 *= -.5 * self.p.scale ** (-2)
        p2_np2 = act_z.transpose(-1, -2).matmul(act_z).unsqueeze(1)
        p2_np2 *= -.5 * self.p.obs_scale ** (-2)
        p_np2 = add_diagonal(p2_np2, -.5 * self.p.prior_scale ** (-2))

        # (num_samples, dim_out, dim_in, dim_in).
        np2 = p_np2 + act_z_.transpose(-1, -2).matmul(t_np2_).matmul(act_z_)

        # Compute mean and covariance matrix for each column of weights.
        # TODO: currently performs two cholesky inversions when only one is
        #  needed.
        prec = -2. * np2
        cov = psd_inverse(prec)
        loc = cov.matmul(np1).squeeze(-1)
        qw = torch.distributions.MultivariateNormal(loc, cov)

        return qw
