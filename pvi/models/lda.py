import torch

from torch import distributions, nn, optim
from .base import Model


class LatentDirichletAllocation(Model, nn.Module):
    """
    Latent Dirichlet allocation model.
    """
    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

    def get_default_nat_params(self):
        """
        :return: {
            "dl_np": Natural parameters of Dirichlet distribution,
            "mn_np": Natural parameters of multinomial distribution,
        }.
        """
        return {
            "dl_np": {
                "np1": [torch.ones(self.hyperparameters["K"])
                        * self.hyperparameters["num_words"][d]
                        / self.hyperparameters["K"]
                        for d in range(self.hyperparameters["D"])],
            },
            "mn_np": {
                "np1": None,
                "np2": [(torch.ones((self.hyperparameters["num_words"][d],
                                     self.hyperparameters["K"]))
                         / self.hyperparameters["K"]).log()
                        for d in range(self.hyperparameters["D"])],
            }
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None,  # Number of documents.
            "K": None,  # Number of topics.
            "V": None,  # Vocabulary size.
            "num_words": None,  # Number of words in each document.
        }

    def elbo(self, data, qtheta, qz, ptheta, pz):
        """
        Returns the ELBO (up to an additive constant) of the latent Dirichlet
        allocation model under q(θ, z), with prior p(θ, z).
        :param data: The local data.
        :param qtheta: The current global posterior q(θ).
        :param qz: The current global posterior q(z).
        :param ptheta: The prior p(θ) (could be cavity).
        :param pz: The prior p(z) (Could be cavity).
        :return: The evidence lower bound.
        """

