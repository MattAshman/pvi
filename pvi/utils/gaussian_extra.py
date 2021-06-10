import torch
import pdb

from pvi.utils.psd_utils import psd_inverse


def joint_from_marginal_lingauss(
    qa, qb, kab, kbb, kaa=None, ikaa=None, b_then_a=False
):
    """
    Computes the joint distribution q(a, b) = q(a) q(b | a).
    :param qa: The marginal distribution, q(a).
    :param kab: K(za, zb).
    :param kbb: K(zb, zb).
    :param kaa: K(za, za).
    :param ikaa: K(za, za)^{-1}.
    :param b_then_a: Order as q(b, a), rather than q(a, b). # TODO
    :return: q(a, b) = q(b | a) q(a).
    """
    dima = kab.shape[-2]
    dimb = kab.shape[-1]
    dimab = dima + dimb
    qa_loc = qa.std_params["loc"]
    qa_cov = qa.std_params["covariance_matrix"]
    qb_loc = qb.std_params["loc"]
    qb_cov = qb.std_params["covariance_matrix"]

    if ikaa is None:
        if kaa is None:
            raise ValueError("Must specifiy either kaa or ikaa.")
        else:
            ikaa = psd_inverse(kaa)

    a = kab.transpose(-1, -2).matmul(ikaa)
    c = kbb - a.matmul(kab)
    ic = psd_inverse(c)
    icat = ic.matmul(a)
    A = qb_cov.matmul(icat)
    At = A.transpose(-1, -2)
    qab_loc = torch.zeros(*kbb.shape[:-2], dimab)
    qab_cov = torch.zeros(*kbb.shape[:-2], dimab, dimab)

    if b_then_a:
        # TODO
        pass
    else:
        qab_loc[..., :dima] = qa_loc
        qab_loc[..., dima:] = (
            A.matmul(qa_loc.unsqueeze(-1)).squeeze(-1) + qb_loc
        )
        qab_cov[..., :dima, :dima] = qa_cov
        qab_cov[..., :dima, dima:] = qa_cov.matmul(At)
        qab_cov[..., dima:, :dima] = A.matmul(qa_cov)
        qab_cov[..., dima:, dima:] = qb_cov + A.matmul(qa_cov).matmul(At)

    assert torch.isclose(qab_cov, qab_cov.transpose(-1, -2)).all()

    return qab_loc, qab_cov
