import torch

from gpytorch.kernels import Kernel as GPyTorchKernel
from gpytorch.settings import trace_mode
from gpytorch.functions import RBFCovariance
from gpytorch.lazy import delazify
from gpytorch.constraints import Positive


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class BayesianKernel(GPyTorchKernel):
    """
    A wrapper for the GPyTorch RBFKernel, which allows a lengthscale to be set
    manually (to retain gradients when learning q(ε)).
    """

    has_manual_lengthscale = False

    def __init__(
        self,
        ard_num_dims=None,
        batch_shape=torch.Size([]),
        active_dims=None,
        lengthscale_prior=None,
        lengthscale_constraint=None,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(
            ard_num_dims, batch_shape, active_dims, lengthscale_prior,
            lengthscale_constraint, eps=eps, **kwargs)

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if self.has_manual_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_buffer(
                "raw_lengthscale",
                torch.zeros(*self.batch_shape, 1, lengthscale_num_dims),
            )
            if lengthscale_prior is not None:
                self.register_prior(
                    "lengthscale_prior",
                    lengthscale_prior,
                    lambda: self.lengthscale,
                    lambda v: self._set_lengthscale(v)
                )

            self.raw_lengthscale_constraint = lengthscale_constraint

    @property
    def dtype(self):
        if self.has_manual_lengthscale:
            return self.lengthscale.dtype
        else:
            for param in self.parameters():
                return param.dtype
            return torch.get_default_dtype()

    @property
    def is_stationary(self) -> bool:
        """
        Property to indicate whether kernel is stationary or not.
        """
        return self.has_lengthscale or self.has_manual_lengthscale

    @property
    def lengthscale(self):
        if self.has_manual_lengthscale:
            return self.raw_lengthscale_constraint.transform(
                self.raw_lengthscale)
        else:
            return None

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def _set_lengthscale(self, value):
        if not self.has_manual_lengthscale:
            raise RuntimeError("Kernel has no lengthscale.")

        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)

        # Inverse transform and constrain.
        raw_lengthscale = self.raw_lengthscale_constraint.inverse_transform(
                    value)

        # Not a parameter, manually set lengthscale.
        self.raw_lengthscale = raw_lengthscale.reshape(
            self.raw_lengthscale.shape)


class BayesianRBFKernel(BayesianKernel):
    """
    A wrapper for the GPyTorch RBFKernel, which allows a lengthscale to be set
    manually (to retain gradients when learning q(ε)).
    """

    has_manual_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
        return RBFCovariance().apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
            ),
        )


class BayesianScaleKernel(BayesianKernel):
    """
    A wrapper for the GPyTorch RBFKernel, which allows an outputscale to be set
    manually (to retain gradients when learning q(ε)).
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel, outputscale_prior=None,
                 outputscale_constraint=None, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super().__init__(**kwargs)
        if outputscale_constraint is None:
            outputscale_constraint = Positive()

        self.base_kernel = base_kernel
        outputscale = torch.zeros(*self.batch_shape) if len(
            self.batch_shape) else torch.tensor(0.0)
        self.register_buffer(name="raw_outputscale",
                             tensor=outputscale)

        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", outputscale_prior,
                lambda: self.outputscale, lambda v: self._set_outputscale(v)
            )

        self.raw_outputscale_constraint = outputscale_constraint

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)

        raw_outputscale = self.raw_outputscale_constraint.inverse_transform(
                value)

        self.raw_outputscale = raw_outputscale.reshape(
            self.raw_outputscale.shape)

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag,
                                               last_dim_is_batch=last_dim_is_batch,
                                               **params)
        outputscales = self.outputscale
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return delazify(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
