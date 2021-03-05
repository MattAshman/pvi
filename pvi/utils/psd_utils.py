import torch

__all__ = ["psd_logdet", "psd_inverse", "add_diagonal"]


def psd_inverse(x=None, chol=None):
    """
    Returns the inverse of positive definite matrix x.
    :param x: Positive definite matrix, shape (*, D, D).
    :param chol: Cholesky factor of s, shape (*, D, D).
    :return: Inverse of x.
    """
    if chol is None:
        chol = torch.cholesky(x)
        num_dims = len(x.shape)
    else:
        num_dims = len(chol.shape)

    if num_dims == 2:
        inverse = torch.cholesky_inverse(chol)
    elif num_dims == 3:
        inverse = torch.stack([torch.cholesky_inverse(l) for l in chol])
    else:
        raise ValueError("x must be either (batch_size, D, D) or (D, D) "
                         "positive definite matrix.")

    return inverse


def psd_logdet(x=None, chol=None):
    """
    Returns the log-determinant of positive definite matrix x.
    :param x: Positive definite matrix, shape (*, D, D).
    :param chol: Cholesky factor of s, shape (*, D, D).
    :return: Log-determinant of x.
    """
    if chol is None:
        chol = torch.cholesky(x)
        num_dims = len(x.shape)
    else:
        num_dims = len(chol.shape)

    if num_dims == 2:
        logdet = 2 * chol.diag().log().sum(-1)
    elif num_dims == 3:
        logdet = torch.stack([2 * l.diag().log().sum(-1) for l in chol])
    else:
        raise ValueError("x must be either (batch_size, D, D) or (D, D) "
                         "positive definite matrix.")

    return logdet


def add_diagonal(x, val):
    """
    Adds val to the diagonal of x.
    :param x: Matrix, shape (*, D, D).
    :param val: Value to add to diagonal of x.
    :return: x + diag(val).
    """
    assert x.shape[-2] == x.shape[-1], "x must be square."

    d = (torch.ones(x.shape[-2], device=x.device) * val).diag_embed()

    return x + d
