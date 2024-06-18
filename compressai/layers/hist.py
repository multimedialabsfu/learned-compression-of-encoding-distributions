import math

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


def histogram(
    x: Tensor,
    bin_centers: Tensor,
    bandwidth: Union[float, Tensor] = 1.0,
    bin_width: Union[float, Tensor] = 1.0,
    kernel: str = "triangular",
    mass: bool = False,
    hard: bool = False,
) -> Tensor:
    """Estimate the histogram in a differentiable way.

    .. note::
        Currently, only uniform histograms are supported.

    Args:
        x: Input tensor.
        bin_centers: Centers of the bins.
        bandwidth: Bandwidth(s) for the kernel function.
        bin_width: Width(s) of the uniformly-sized bins.
            Required if ``hard=True``.
        kernel: Kernel for kernel density estimation
            ``(`uniform`, `triangular`, `epanechnikov`, `gaussian`)``.
        mass: Return probability mass histogram.
        hard: Return exact hard histogram, but pass through soft gradients.
    """
    # x.shape == (*other_dims, num_samples)
    # bin_centers.shape == (num_bins,)
    # u.shape == (*other_dims, num_samples, num_bins)
    u = (x[..., None] - bin_centers) / bandwidth

    # https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use
    if kernel == "uniform":
        # WARNING: No gradients are passed through for this kernel.
        y = (u.detach().abs() <= 0.5).to(u.dtype)
    elif kernel == "triangular":
        y = (1.0 - u.abs()).clip(min=0)
    elif kernel == "epanechnikov":
        y = (1.0 - u**2).clip(min=0)
    elif kernel == "gaussian":
        y = (-0.5 * u**2).exp()
    else:
        raise ValueError(f"Unknown kernel {kernel}.")

    if hard:
        y_hard = (u.detach().abs() <= 0.5 * bin_width / bandwidth).to(u.dtype)
        y = y - y.detach() + y_hard

    hist = y.sum(dim=-2)

    if mass:
        hist = hist / x.shape[-1]
        # hist = hist / (hist.sum(axis=-1, keepdim=True) + eps)

    return hist


def uniform_histogram(
    x: Tensor,
    num_bins: int = 256,
    bandwidth: Union[float, Tensor] = 1.0,
    bin_width: Union[float, Tensor] = 1.0,
    kernel: str = "triangular",
    mass: bool = False,
    hard: bool = False,
) -> Tensor:
    """Estimate the histogram in a differentiable way.

    Args:
        x: Input tensor. Must be in the range ``[0, num_bins - 1)``.
        num_bins: Number of bins.
        bandwidth: Bandwidth(s) for the kernel function.
        bin_width: Width(s) of the uniformly-sized bins.
            Required if ``hard=True``.
        kernel: Kernel for kernel density estimation
            ``(`uniform`, `triangular`, `epanechnikov`, `gaussian`)``.
        mass: Return probability mass histogram.
        hard: Return exact hard histogram, but pass through soft gradients.
    """
    # Only tested for the following settings.
    assert bandwidth == 1.0
    assert bin_width == 1.0
    assert kernel == "triangular" or kernel == "epanechnikov"

    # x.shape == (*other_dims, num_samples)
    # bin_centers.shape == (*other_dims, num_samples, num_nearest_bins)
    # u.shape == (*other_dims, num_samples, num_nearest_bins)
    # hist.shape == (*other_dims, num_bins)

    num_nearest_bins = 2
    bin_offsets = torch.arange(num_nearest_bins, device=x.device, dtype=x.dtype)
    bin_centers = x.floor()[..., None] + bin_offsets
    indexes = bin_centers.long()

    u = (x[..., None] - bin_centers) / bandwidth

    # https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use
    if kernel == "uniform":
        # WARNING: No gradients are passed through for this kernel.
        y = (u.detach().abs() <= 0.5).to(u.dtype)
    elif kernel == "triangular":
        y = (1.0 - u.abs()).clip(min=0)
    elif kernel == "epanechnikov":
        y = (1.0 - u**2).clip(min=0)
    elif kernel == "gaussian":
        y = (-0.5 * u**2).exp()
    else:
        raise ValueError(f"Unknown kernel {kernel}.")

    if hard:
        y_hard = (u.detach().abs() <= 0.5 * bin_width / bandwidth).to(u.dtype)
        y = y - y.detach() + y_hard

    shape = (*x.shape[:-1], -1)
    hist = x.new_zeros((*x.shape[:-1], num_bins))
    hist.scatter_add_(dim=-1, index=indexes.reshape(shape), src=y.reshape(shape))

    if mass:
        hist = hist / x.shape[-1]
        # hist = hist / (hist.sum(axis=-1, keepdim=True) + eps)

    return hist


class UniformHistogram(nn.Module):
    def __init__(self, num_bins=256, min=0.0, max=255.0, **kwargs):
        super().__init__()
        self.num_bins = num_bins
        self.min = min
        self.max = max
        self.bin_width = (max - min) / (num_bins - 1)
        self.register_buffer("bin_centers", torch.linspace(min, max, num_bins))
        self.kwargs = kwargs

    def forward(self, x, **kwargs):
        return uniform_histogram(
            x - self.min,
            num_bins=self.num_bins,
            bin_width=self.bin_width,
            **{**self.kwargs, **kwargs},
        )
        # return histogram(
        #     x, self.bin_centers, bin_width=self.bin_width, **{**self.kwargs, **kwargs}
        # )


# From https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml:
#
# - name: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor  # noqa: E501
#   self: gather_backward(grad, self, dim, index, sparse_grad)
#   index: non_differentiable
#   result: auto_linear
#
# - name: scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
#   self: grad
#   index: non_differentiable
#   src: grad.gather(dim, index)
#   result: scatter_add(self_t, dim, index, src_t)
#
class DiscreteIndexingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, x, diff_kernel, grad_f_multiplier=1.0):
        assert x.shape[:-1] == f.shape[:-1]
        assert x.min() >= 0
        assert x.max() <= f.shape[-1] - 1
        grad_f_multiplier = f.new_tensor(grad_f_multiplier)
        ctx.save_for_backward(f, x, diff_kernel, grad_f_multiplier)
        return DiscreteIndexingFunction._lerp(f, x)

    @staticmethod
    def backward(ctx, grad_output):
        f, x, diff_kernel, grad_f_multiplier = ctx.saved_tensors
        df_dx = DiscreteIndexingFunction._estimate_derivative(f, diff_kernel)
        df_dx_at_x = DiscreteIndexingFunction._lerp(df_dx, x)
        grad_x = df_dx_at_x * grad_output
        grad_f = DiscreteIndexingFunction._dout_df(f, x, grad_output)
        grad_f = grad_f * grad_f_multiplier
        return grad_f, grad_x, None, None

    @staticmethod
    def _lerp(f, x):
        x1 = x.floor().long()
        x2 = x1 + 1
        y1 = f.gather(dim=-1, index=x1)
        y2 = f.gather(dim=-1, index=x2)
        dx = x - x1
        return y1 * (1 - dx) + y2 * dx

    @staticmethod
    def _estimate_derivative(f, diff_kernel):
        # Pad f, then estimate derivative via finite difference kernel.
        *other_dims, num_bins = f.shape
        pad_width = diff_kernel.shape[-1] // 2
        f = f.reshape(math.prod(other_dims), 1, num_bins)
        f = F.pad(f, pad=(pad_width, pad_width), mode="replicate")
        df_dx = F.conv1d(f, weight=diff_kernel).reshape(*other_dims, num_bins)
        return df_dx

    @staticmethod
    def _dout_df(f, x, grad_output):
        # Nothing fancy; just manually compute the standard derivative.
        x1 = x.floor().long()
        x2 = x1 + 1
        dx = x - x1
        grad_y1 = grad_output * (1 - dx)
        grad_y2 = grad_output * dx
        grad_f = torch.zeros_like(f)
        grad_f.scatter_add_(dim=-1, index=x1, src=grad_y1)
        grad_f.scatter_add_(dim=-1, index=x2, src=grad_y2)
        return grad_f


class DiscreteIndexing(nn.Module):
    def __init__(self, grad_f_multiplier=1.0):
        super().__init__()
        self.register_buffer("diff_kernel", self._get_diff_kernel())
        self.grad_f_multiplier = grad_f_multiplier

    def forward(self, f, x):
        return DiscreteIndexingFunction.apply(
            f, x, self.diff_kernel, self.grad_f_multiplier
        )

    def _get_diff_kernel(self):
        smoothing_kernel = torch.tensor([[[0.25, 0.5, 0.25]]])
        difference_kernel = torch.tensor([[[-0.5, 0, 0.5]]])
        return F.conv1d(
            smoothing_kernel,
            difference_kernel.flip(-1),
            padding=difference_kernel.shape[-1] - 1,
        )


# def cumulative_histogram(...):
#     pass
#     # WARN: Not the below... that's dc/dx...
#     # not how the ith bin changes as y_j changes, i.e. dc_i/dy_j...!!!!!
#     #
#     # CDF variant: estimate derivative using finite differences + epsilon.
#     # Ensure centered around correct bin...
#     # and that pdf = cdf[1:] - cdf[:-1] works out for the right +/- 1/2.
#     # Just make the hard pdf via fast histogram technique. cumsum for hard cdf.
#
#     # Actually:
#     #
#     # Similar procedure to histogram, but with a different kernel.
#     # Probably need a "unit step" "kernel" function!


# TODO Penalize out-of-bounds. And clip, too?

# TODO try this first, then the various alternative schemes
# (including piecewise differentiable).


# TODO
#
# - Haven't dealt with channels axis for bin_centers yet...? Or have we (sort of)?
# - Do we need to directly handle out-of-bounds, or can we just pre-clip x?
#
# bins: int = 256,
# range: Optional[Tuple[float, float]] = None,
# eps: float = 1e-10,


# NOTE: pmf == pdf for unit width bins, according to NumPy:
# https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
