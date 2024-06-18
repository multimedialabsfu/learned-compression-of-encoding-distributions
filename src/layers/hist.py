from typing import Union

import torch
import torch.nn as nn

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
