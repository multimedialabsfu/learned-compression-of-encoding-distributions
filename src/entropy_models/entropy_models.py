import math
from typing import Any

import torch

from compressai.entropy_models import EntropyBottleneck


class EntropyBottleneckExtended(EntropyBottleneck):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def update(self, force: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        self._update_quantiles()

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]

        pmf, lower, upper = self._likelihood(samples, stop_gradient=True)
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    @torch.no_grad()
    def _update_quantiles(self, search_radius=1e4, rtol=1e-4, atol=1e-3):
        device = self.quantiles.device
        shape = (self.channels, 1, 1)
        low = torch.full(shape, -search_radius, device=device)
        high = torch.full(shape, search_radius, device=device)

        def f(y, self=self):
            return self._logits_cumulative(y, stop_gradient=True)

        for i in range(len(self.target)):
            q_i = self._search_target(f, self.target[i], low, high, rtol, atol)
            self.quantiles[:, :, i] = q_i[:, :, 0]

    @staticmethod
    def _search_target(f, target, low, high, rtol=1e-4, atol=1e-3, strict=False):
        assert (low <= high).all()
        if strict:
            assert ((f(low) <= target) & (target <= f(high))).all()
        else:
            low = torch.where(target <= f(high), low, high)
            high = torch.where(f(low) <= target, high, low)
        while not torch.isclose(low, high, rtol=rtol, atol=atol).all():
            mid = (low + high) / 2
            f_mid = f(mid)
            low = torch.where(f_mid <= target, mid, low)
            high = torch.where(f_mid >= target, mid, high)
        return (low + high) / 2


EntropyBottleneck.update = EntropyBottleneckExtended.update
EntropyBottleneck._update_quantiles = EntropyBottleneckExtended._update_quantiles
EntropyBottleneck._search_target = staticmethod(
    EntropyBottleneckExtended._search_target
)


@torch.no_grad()
def pdf_layout(q: torch.Tensor, method="compact"):
    """Determine discretized PDF layout.

    Returns:
        - q_indexes: The indexes of the quantiles in the discretized PDF.
    """
    assert q.ndim == 2
    left = (q[:, 1] - q[:, 0]).ceil().long()
    right = (q[:, 2] - q[:, 1]).ceil().long()
    zero = torch.zeros_like(left)
    q_indexes = torch.stack([-left, zero, right], axis=1)

    if method == "none":
        pass
    elif method == "compact":
        # Choose ranges such that the minimum range is used.
        q_indexes += left
    elif method == "balanced":
        # Choose ranges such that the median is centered in the discretized PDF.
        # Not "space-efficient", but this is easier to work with.
        q_indexes += max(left.max().item(), right.max().item())
    else:
        raise NotImplementedError

    # NOTE: The probability for a symbol centered at y is for a continuous cdf is:
    # cdf(y + 1/2)[c] - cdf(y - 1/2)[c] == pdf[(y - q[c, 1]).round().long() - offset[c]]

    return q_indexes


@torch.no_grad()
def likelihood_func_to_pdf(
    likelihood_func,
    q,
    q_indexes,
    symbol_boundaries=None,
    oob_symbol_pos=None,
    min_pdf_size=0,
):
    device = q.device
    num_channels = q.shape[0]
    num_oob_symbols = int(bool(oob_symbol_pos))

    if symbol_boundaries is None:
        symbol_boundaries = q_indexes

    assert (symbol_boundaries >= 0).all()

    min_symbol = symbol_boundaries[:, 0]
    max_symbol = symbol_boundaries[:, -1]
    pdf_max_size = max(min_pdf_size, max_symbol.max().item() + 1 + num_oob_symbols)

    symbols = torch.arange(pdf_max_size, device=device)
    y_hat = symbols + (q[:, 1, None] - q_indexes[:, 1, None])
    _, pdf = likelihood_func(y_hat.unsqueeze(0))
    pdf = pdf.squeeze(0)
    assert pdf.shape == (num_channels, pdf_max_size)

    # Clip symbols to be within the PDF range.
    clip_min_mask = symbols >= min_symbol[None, :, None]
    clip_max_mask = symbols <= max_symbol[None, :, None]
    pdf = pdf.where(clip_min_mask & clip_max_mask, torch.zeros_like(pdf))

    remainder = (1 - pdf.sum(axis=-1)).clip(min=0)

    _embed_remainder(pdf, remainder, max_symbol, oob_symbol_pos)

    return pdf, remainder


@torch.no_grad()
def y_to_pdf(y, q, q_indexes, symbol_boundaries=None, oob_symbol_pos="adjacent"):
    n, c, *other_dims = y.shape
    y = y.reshape((n, c, -1))
    symbols = (y - q[:, 1, None]).round().long() + q_indexes[:, 1, None]

    if symbol_boundaries is None:
        symbol_boundaries = q_indexes

    min_symbol = symbol_boundaries[:, 0]
    max_symbol = symbol_boundaries[:, -1]
    pdf_max_size = max_symbol.max().item() + 1 + 1

    if oob_symbol_pos is None:
        raise NotImplementedError
    elif oob_symbol_pos == "adjacent":
        oob_symbol = max_symbol[None, :, None] + 1
    elif oob_symbol_pos == "aligned":
        oob_symbol = pdf_max_size - 1
    else:
        raise NotImplementedError

    # Clip symbols to be within the PDF range.
    clip_min_mask = symbols >= min_symbol[None, :, None]
    clip_max_mask = symbols <= max_symbol[None, :, None]
    symbols = symbols.where(clip_min_mask & clip_max_mask, oob_symbol)

    counts = symbols.new_zeros((n, c, pdf_max_size))
    counts.scatter_add_(axis=-1, index=symbols, src=torch.ones_like(symbols))
    total_count = math.prod(other_dims)
    pdf = counts / total_count

    remainder = _unembed_remainder(pdf, max_symbol, oob_symbol_pos)

    return pdf, remainder


def _embed_remainder(pdf, remainder, max_symbol, oob_symbol_pos):
    if oob_symbol_pos is None:
        pass
    elif oob_symbol_pos == "adjacent":
        pdf[..., max_symbol + 1] = remainder
    elif oob_symbol_pos == "aligned":
        pdf[..., -1] = remainder
    else:
        raise NotImplementedError


def _unembed_remainder(pdf, max_symbol, oob_symbol_pos):
    if oob_symbol_pos is None:
        raise NotImplementedError
    elif oob_symbol_pos == "adjacent":
        return pdf[..., max_symbol + 1]
    elif oob_symbol_pos == "aligned":
        return pdf[..., -1]
    else:
        raise NotImplementedError


def inv_sigmoid(x):
    return x.log() - (1 - x).log()
