import torch

from pvi.utils.psd_utils import psd_inverse


def mvstandard2natural(mu, cov):
    """
    Converts standard parameterisation of a multivariate Gaussian to natural
    parameterisation.
    :param mu: The mean, (*, D).
    :param cov: The covariance matrix, (*, D, D).
    :return: np1, np2, (*, D), (*, D, D).
    """
    assert mu.shape[-1] == cov.shape[-1], "mu (*, D), cov (*, D, D)."
    assert cov.shape[-2] == cov.shape[-1], "cov (*, D, D)."

    prec = psd_inverse(cov)
    np2 = - 0.5 * prec
    np1 = prec.matmul(mu.unsqueeze(-1)).squeeze(-1)

    return np1, np2


def standard2natural(mu, sigma):
    """
    Converts standard parameterisation of a 1-D Gaussian to natural
    parameterisation.
    :param mu: The mean, (*, D).
    :param sigma: The standard deviation, (*, D).
    :return: np1, np2, (*, D), (*, D).
    """
    assert mu.shape[-1] == sigma.shape[-1], "mu (*, D), sigma (*, D)."
    assert len(mu.shape) == len(sigma.shape), "mu (*, D), sigma (*, D)."

    np2 = -0.5 * sigma.pow(-2)
    np1 = sigma.pow(-2) * mu

    return np1, np2
