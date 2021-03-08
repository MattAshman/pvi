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
    np2 = -0.5 * prec
    np1 = prec.matmul(mu.unsqueeze(-1)).squeeze(-1)

    return np1, np2


def mvnatural2standard(np1, np2):
    """
    Converts natural parameterisation of a multivariate Gaussian to standard
    parameterisation.
    :param np1: Natural parameter 1, (*, D).
    :param np2: Natural parameter 2, (*, D, D).
    :return: mu, cov, (*, D), (*, D, D).
    """
    assert np1.shape[-1] == np2.shape[-1], "np1 (*, D), np2 (*, D, D)."
    assert np2.shape[-2] == np2.shape[-1], "np2 (*, D, D)."

    prec = -2 * np2
    cov = psd_inverse(prec)
    mu = cov.matmul(np1.unsqueeze(-1)).squeeze(-2)

    return mu, cov


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

    np2 = -0.5 * sigma ** (-2)
    np1 = sigma ** (-2) * mu

    return np1, np2


def natural2standard(np1, np2):
    """
    Converts natural parameterisation of a multivariate Gaussian to standard
    parameterisation.
    :param np1: Natural parameter 1, (*, D).
    :param np2: Natural parameter 2, (*, D).
    :return: mu, sigma, (*, D), (*, D).
    """
    assert np1.shape[-1] == np2.shape[-1], "np1 (*, D), np2 (*, D)."
    assert len(np1.shape) == len(np2.shape), "np1 (*, D), np2 (*, D)."

    sigma = (-2 * np2) ** (-0.5)
    mu = sigma ** 2 * np1

    return mu, sigma
