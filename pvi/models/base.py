from abc import ABC, abstractmethod


class Model(ABC):
    """
    An abstract class for probabilistic models defined by a likelihood
    p(y | θ, x) and (approximate) posterior q(θ).
    """
    def __init__(self, eps=None, hyperparameters=None):
        # Hyperparameters of the model.
        if hyperparameters is None:
            hyperparameters = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.set_hyperparameters(hyperparameters)

        # Parameters of the model.
        if eps is None:
            eps = {}

        self.eps = {**eps, **self.get_default_eps()}

    @abstractmethod
    def get_default_nat_params(self):
        """
        :return: A default set of natural parameters for the prior.
        """
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    @abstractmethod
    def get_default_hyperparameters(self):
        """
        :return: A default set of hyperparameters for the model.
        """
        raise NotImplementedError

    def set_eps(self, eps):
        self.eps = {**self.eps, **eps}

    @abstractmethod
    def get_default_eps(self):
        """
        :return: A default set of parameters for the model.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x, q):
        """
        Returns the (approximate) predictive posterior.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(θ).
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        raise NotImplementedError

    @abstractmethod
    def likelihood_forward(self, x, theta):
        """
        Returns the model's likelihood p(y | θ, x).
        :param x: The input locations to make predictions at.
        :param theta: The latent variables of the model.
        :return: p(y | θ, x)
        """
        raise NotImplementedError

    def likelihood_log_prob(self, data, theta):
        """
        Compute the log probability of the data under the model's likelihood.
        :param data: The data to compute the log likelihood of.
        :param theta: The latent variables of the model.
        :return: log p(y | x, θ)
        """
        dist = self.likelihood_forward(data["x"], theta)
        return dist.log_prob(data["y"])

    @abstractmethod
    def conjugate_update(self, data, q, t=None):
        """
        If the likelihood is conjugate with q(θ), performs a conjugate update.
        :param data: The data to compute the conjugate update with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: The updated posterior, p(θ | data).
        """
        raise NotImplementedError

    @abstractmethod
    def expected_log_likelihood(self, data, q):
        """
        Computes the expected log likelihood of the data under q(θ).
        :param data: The data to compute the conjugate update with.
        :param q: The current global posterior q(θ).
        :return: ∫ q(θ) log p(y | x, θ) dθ.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def conjugate_family(self):
        raise NotImplementedError
