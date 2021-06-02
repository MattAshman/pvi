import torch

from pvi.models.bnn.bnn import FullyConnectedBNN


class FullyConnectedIPBNN(FullyConnectedBNN):
    """
    Bayesian neural network with inducing point variational approximation.
    """

    def sample_weights(self, q, log_probs=False, p=None, num_samples=1):
        """
        Samples q(w_l) at each layer, propagating the inducing locations
        forward each layer.
        :param q: The approximate posterior distribution. Also contains
        incuding locations.
        :param log_probs: Whether to return the log-probabilities also.
        :param p: The prior distribution.
        :param num_samples: Number of samples.
        :return: {w_l}.
        """
        z = q.inducing_locations

        theta = []
        log_qw = []
        log_pw = []
        # (num_samples, m, dim_in - 1).
        z = torch.cat(num_samples * [z.unsqueeze(0)])
        for i in range(len(self.shapes)):
            z = torch.cat([z, torch.ones((*z.shape[:-1], 1))], dim=-1)

            # Compute the distribution over weights and biases at this layer.
            qw = q.compute_dist(i, z)

            # Sample the weights and biases.
            # (num_samples, dim_in, dim_out).
            w = qw.rsample().transpose(-1, -2)
            theta.append(w)

            # Evaluate log-probabilities if needed.
            if log_probs:
                log_qw.append(qw.log_prob(w.transpose(-1, -2)).sum(-1))

                if p is not None:
                    log_pw.append(p.log_prob(layer=i, act_z=z, theta=w))
                    # pw = p.compute_dist(i, z)
                    # log_pw.append(pw.log_prob(w.transpose(-1, -2)).sum(-1))

            # Propagate inducing locations forward.
            # z = self.activation(z).matmul(w)
            z = z.matmul(w)
            if i < len(self.shapes) - 1:
                z = self.activation(z)

        if log_probs:
            if p is not None:
                return theta, log_pw, log_qw
            else:
                return theta, log_qw

        return theta

    def forward(self, x, q, **kwargs):
        thetas = self.sample_weights(
            q, num_samples=self.config["num_predictive_samples"])

        qy = self.likelihood_forward(x, thetas, samples_first=False)

        mix = torch.distributions.Categorical(
            logits=torch.ones(size=qy.batch_shape))
        qy = torch.distributions.MixtureSameFamily(mix, qy)

        return qy

    def likelihood_forward(self, x, thetas, samples_first=True):
        assert len(x.shape) in [1, 2], "x must be (N, D)."

        if len(x.shape) == 1:
            x = x[None, :]

        for i, theta in enumerate(thetas):
            assert len(theta.shape) in [2, 3], "theta must be (S, Din, Dout)."
            if len(theta.shape) == 2:
                thetas[i] = theta[None, :, :]

        for i, w in enumerate(thetas):
            # Bias term.
            x = torch.cat([x, torch.ones((*x.shape[:-1], 1))], dim=-1)
            # x = self.activation(x).matmul(w)
            x = x.matmul(w)

            # Don't apply ReLU to final layer.
            if i < len(thetas) - 1:
                x = self.activation(x)

        return self.pred_dist_from_tensor(x, samples_first=samples_first)

    def elbo(self, data, p, q, num_samples=1):
        """
        Computes the ELBO of the data under q(θ). Different to standard
        computation, as can no longer compute the KL-divergence in closed-form.
        :param data: The data to compute the conjugate update with.
        :param p: The current prior p(θ).
        :param q: The current posterior q(θ).
        :param num_samples: The number of samples to estimate the expected
        log-likelihood with.
        :param n: The number of training datapoints, potentially different to
        batch size, len(data).
        :return: ∫ q(θ) log p(y | x, θ) dθ.
        """
        x = data["x"]
        y = data["y"]

        theta, log_pw, log_qw = self.sample_weights(
            q, log_probs=True, p=p, num_samples=num_samples)
        kl = (sum(log_qw) - sum(log_pw)).mean()

        # Propagate inputs forward.
        for i, w in enumerate(theta):
            x = torch.cat([x, torch.ones((*x.shape[:-1], 1))], dim=-1)
            x = x.matmul(w)

            if i < len(theta) - 1:
                x = self.activation(x)

        qy = self.pred_dist_from_tensor(x, samples_first=True)
        ll = qy.log_prob(y).squeeze(-1).sum(-1).mean()

        return ll, kl


class RegressionIPBNN(FullyConnectedIPBNN):

    def forward(self, x, q, **kwargs):
        thetas = self.sample_weights(
            q, num_samples=self.config["num_predictive_samples"])

        # (S, N, output_dim)
        qy = self.likelihood_forward(x, thetas, samples_first=False)

        return qy

    def pred_dist_from_tensor(self, tensor, samples_first=True):
        # (S, N, output_dim).
        loc = tensor[:, :, :self.config["output_dim"]]
        scale = torch.exp(tensor[:, :, self.config["output_dim"]:])

        return torch.distributions.Normal(loc=loc, scale=scale)

    @property
    def shapes(self):
        shapes = []
        for i in range(self.config["num_layers"] + 1):
            if i == 0:
                shapes.append((self.config["D"] + 1,
                               self.config["latent_dim"]))
            elif i == self.config["num_layers"]:
                # Weight matrix.
                shapes.append((self.config["latent_dim"] + 1,
                               self.config["output_dim"] * 2))
            else:
                # Weight matrix.
                shapes.append((self.config["latent_dim"] + 1,
                               self.config["latent_dim"]))

        return shapes


class ClassificationIPBNN(FullyConnectedIPBNN):

    def pred_dist_from_tensor(self, tensor, samples_first=True):
        if not samples_first:
            tensor = torch.transpose(tensor, 0, 1)

        return torch.distributions.Categorical(logits=tensor)
