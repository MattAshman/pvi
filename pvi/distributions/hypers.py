class HyperparameterDistribution:
    """
    Maintains the distributions over hyperparameters.
    """
    def __init__(self, distributions=None):
        """
        :param distributions: A dictionary of (hyperparameter, distribution)
        pairs.
        """
        self.distributions = distributions

    def non_trainable_copy(self):
        return type(self)(
            distributions={
                k: v.non_trainable_copy()
                for k, v in self.distributions.items()
            }
        )

    def trainable_copy(self):
        return type(self)(
            distributions={
                k: v.trainable_copy()
                for k, v in self.distributions.items()
            }
        )

    def kl_divergence(self, other):
        return {k: v.kl_divergence(other.distributions[k])
                for k, v in self.distributions.items()}

    def log_prob(self, args_dict, kwargs_dict):
        return {k: v.log_prob(**args_dict[k], **kwargs_dict[k])
                for k, v in self.distributions.items()}

    def sample(self, *args, **kwargs):
        return {k: v.sample(*args, **kwargs)
                for k, v in self.distributions.items()}

    def rsample(self, *args, **kwargs):
        return {k: v.rsample(*args, **kwargs)
                for k, v in self.distributions.items()}

    def parameters(self):
        parameters = [list(v.parameters()) for v in self.distributions.values()]
        return [item for sublist in parameters for item in sublist]


class HyperparameterFactor:
    """
    Maintains the factors over hyperparameters.
    """
    def __init__(self, factors=None):
        """
        :param factors: A dictionary of (hyperparameter, factor)
        pairs.
        """
        self.factors = factors

    def compute_refined_factor(self, q1, q2):
        return type(self)(
            factors={
                k: v.compute_refined_factor(q1[k], q2[k])
                for k, v in self.factors.items()
            }
        )

    def __call__(self, thetas):
        return {k: v(thetas[k]) for k, v in self.factors.items()}

    def log_h(self, thetas):
        return {k: v.log_h(thetas[k]) for k, v in self.factors.items()}

    def npf(self, thetas):
        return {k: v.npf(thetas[k]) for k, v in self.factors.items()}

    def nat_from_dist(self, q):
        return {k: v.nat_from_dist(q[k]) for k, v in self.factors.items()}

    def dist_from_nat(self, np):
        return {k: v.dist_from_nat(np[k]) for k, v in self.factors.items()}
