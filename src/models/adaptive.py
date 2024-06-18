from contextlib import contextmanager

import torch
import torch.nn as nn
from compressai.entropy_models.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.models.base import CompressionModel
from compressai.models.google import (
    FactorizedPrior,
    MeanScaleHyperprior,
    ScaleHyperprior,
)
from compressai.registry import register_model, register_module
from compressai.registry.torch import MODELS

from src.entropy_models.entropy_models import pdf_layout
from src.layers import UniformHistogram
from src.ops import RangeBound


class FreezeMixin:
    def _freeze_modules(self, freeze_modules, unfreeze_modules):
        # Disable gradients for specified modules.
        if "all" in freeze_modules:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for module_name in freeze_modules:
                for param in getattr(self, module_name).parameters():
                    param.requires_grad = False

        # Reenable gradients for specified modules.
        if "all" in unfreeze_modules:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for module_name in unfreeze_modules:
                for param in getattr(self, module_name).parameters():
                    param.requires_grad = True


@register_module("AdaptiveLatentCodec")
class AdaptiveLatentCodec(LatentCodec):
    def __init__(
        self,
        entropy_bottleneck: EntropyBottleneck,
        pdf_model: CompressionModel,
        enable_adaptive=True,
        num_bins=256,
        bound_factor=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.enable_adaptive = enable_adaptive

        self._unregistered_modules = {
            "entropy_bottleneck": entropy_bottleneck,
            "pdf_model": pdf_model,
            # "histogram": histogram,
            # "bound": bound,
        }

        # assert M == pdf_model_conf["hp"]["C"]
        # self.pdf_model = MODELS[pdf_model_conf["name"]](**pdf_model_conf["hp"])

        y_min = -(num_bins // 2)
        self.histogram = UniformHistogram(
            num_bins=num_bins,
            min=y_min,
            max=y_min + num_bins - 1,
        )

        # self.discrete_indexing = DiscreteIndexing()

        # Recommendation: set y bounds conservatively to 75% of histogram range.
        bound_radius = (self.histogram.max - self.histogram.min) / 2 * bound_factor
        bound_center = (self.histogram.max + self.histogram.min) / 2
        self.bound = RangeBound(
            min=bound_center - bound_radius,
            max=bound_center + bound_radius,
        )

        # self.y_offset = nn.Parameter(torch.zeros((M,)))

    @property
    def entropy_bottleneck(self):
        return self._unregistered_modules["entropy_bottleneck"]

    @property
    def pdf_model(self):
        return self._unregistered_modules["pdf_model"]

    # @property
    # def histogram(self):
    #     return self._unregistered_modules["histogram"]

    # @property
    # def bound(self):
    #     return self._unregistered_modules["bound"]

    # @property
    # def discrete_indexing(self):
    #     return self._unregistered_modules["discrete_indexing"]

    @property
    def y_offset(self):
        q = self.entropy_bottleneck.quantiles.squeeze(1)
        return q[:, 1]

    def forward(self, y):
        n, c, h, w = y.shape

        if self.enable_adaptive:
            y_hat, _ = self.entropy_bottleneck(y)

            pdf_input, y = self._prepare_pdf_input(y, hard=True)
            pdf_out = self.pdf_model(pdf_input)
            pdf_hat = pdf_out["x_hat"]
            pdf_likelihoods = pdf_out["likelihoods"]

            pdf_actual = pdf_input[:, 0]
            chan_rate_loss = cross_entropy(pdf_actual, pdf_hat, axis=-1)
            assert chan_rate_loss.shape == (n, c)
            y_likelihoods = (2**-chan_rate_loss)[..., None, None].repeat(1, 1, h, w)
            assert y_likelihoods.shape == y.shape
        else:
            y_hat, y_likelihoods = self.entropy_bottleneck(y)

            # For visualization/analysis only.
            pdf_input, _ = self._prepare_pdf_input(
                y, hard=True, input_format=["pdf", "pdf_default"]
            )
            pdf_actual = pdf_input[:, 0]
            pdf_hat = pdf_input[:, 1]
            pdf_likelihoods = {}

        return {
            "y_hat": y_hat,
            "likelihoods": {
                "y": y_likelihoods,
                **{f"pdf_{k}": v for k, v in pdf_likelihoods.items()},
            },
            "p": pdf_actual,
            "p_hat": pdf_hat,
        }

    def compress(self, y):
        if not self.enable_adaptive:
            y_strings = self.entropy_bottleneck.compress(y)
            y_hat = self.entropy_bottleneck.decompress(y_strings, y.size()[-2:])

            # For visualization/analysis only.
            pdf_input, _ = self._prepare_pdf_input(
                y, hard=True, input_format=["pdf", "pdf_default"]
            )
            # pdf_actual = pdf_input[:, 0]
            pdf_hat = pdf_input[:, 1]

            return {
                "strings": [y_strings],
                "shape": {"y": y.shape[-2:]},
                "y_hat": y_hat,
                "pdf_x_default": None,
                "p_hat": pdf_hat,
            }

        batch_size, *_ = y.shape

        # TODO why q_indexes?

        q = self.entropy_bottleneck.quantiles.squeeze(1)
        q_indexes = pdf_layout(q, method="none") - self.histogram.min

        # NOTE Not really necessary but oh well
        y_offset = self.y_offset[None, :, None, None]
        y = (y - y_offset).round() + y_offset

        # assert (
        #     self._prepare_pdf_input(y, hard=True)[0]
        #     - self._prepare_pdf_input(y, hard=False)[0]
        # ).abs().mean() <= 1e-4

        pdf_input, y = self._prepare_pdf_input(y, hard=True)
        pdf_out_enc = self.pdf_model.compress(pdf_input)
        pdf_out_dec = self.pdf_model.decompress(**pdf_out_enc)
        pdf_hat = pdf_out_dec["x_hat"]

        y_strings = []
        y_hat_ = []

        for n in range(batch_size):
            y_hat_n = self.rdoq(y[n].unsqueeze(0))
            y_hat_.append(y_hat_n)
            with self._override_eb(**self._compute_eb(pdf_hat[n], q_indexes)) as eb:
                [y_string] = eb.compress(y_hat_n)
            assert isinstance(y_string, bytes)
            y_strings.append(y_string)

        y_hat = torch.cat(y_hat_, dim=0)

        return {
            "strings": [
                y_strings,
                *pdf_out_enc["strings"],
            ],
            "shape": {
                "y": y.shape[-2:],
                "pdf": pdf_out_enc["shape"],
            },
            "pdf_x_default": pdf_out_enc["x_default"],
            # Additional outputs:
            "y_hat": y_hat,
            "p_hat": pdf_hat,
        }

    def decompress(self, strings, shape, pdf_x_default, **kwargs):
        if not self.enable_adaptive:
            [y_strings] = strings
            y_hat = self.entropy_bottleneck.decompress(y_strings, shape["y"])
            return {"y_hat": y_hat}

        assert len(strings) == 2
        [y_strings, *pdf_strings] = strings

        # TODO why q_indexes?
        q = self.entropy_bottleneck.quantiles.squeeze(1)
        q_indexes = pdf_layout(q, method="none") - self.histogram.min

        pdf_out_dec = self.pdf_model.decompress(
            pdf_strings, shape["pdf"], pdf_x_default
        )
        pdf_hat = pdf_out_dec["x_hat"]
        batch_size, *_ = pdf_hat.shape

        y_hat_ = []

        for n in range(batch_size):
            with self._override_eb(**self._compute_eb(pdf_hat[n], q_indexes)) as eb:
                y_hat_n = eb.decompress([y_strings[n]], shape["y"])
            y_hat_.append(y_hat_n)

        y_hat = torch.cat(y_hat_, dim=0)

        return {
            "y_hat": y_hat,
        }

    def _prepare_pdf_input(self, y, hard=True, input_format=None):
        n, *_ = y.shape
        q = self.entropy_bottleneck.quantiles.squeeze(1)

        # Clip y within the range of the histogram.
        # Another valid alternative is to bound the likelihood function
        # via a sharp regularization term, and hope that y tends towards
        # the required range; however, since OOB events are unlikely,
        # bounding y earlier may be a more effective method. Naturally,
        # one can use both methods together for extra safety.
        y_offset = self.y_offset[None, :, None, None]
        y_centered = self.bound(y - y_offset)
        y = y_centered + y_offset

        # TODO Can we remove the centering? It leads to distribution shifting, no?
        # Or maybe that's OK, since we're only interested in bounding
        # the newly centered signal, no longer the original "y".
        # BUT... we can't measure the center using conventional aux
        # loss... since the pdf_default is not trained! Need some other
        # way to enforce, e.g. measure how many elements go outside
        # bounds, and then enforce some gradients on them... and
        # encourage g_a not to generate such data, too.
        #
        # ...Or perhaps just including a separate "bias" parameter will do. Of course.

        inputs = {}
        input_format = input_format or self.pdf_model.input_format

        if any(k in input_format for k in ("pdf_default", "cdf_default")):
            # NOTE: training=False disables quantization error simulation (via noise).
            # Since pdf_default is derived from an "exact" source, we disable noise.
            y_default = self.histogram.bin_centers[None, None, :] + q[None, :, None, 1]
            _, pdf_default = self.entropy_bottleneck(y_default, training=False)
            pdf_default = pdf_default.repeat(n, 1, 1)
            # TODO Should we detach the pdf_default?
            # TODO Doesn't the + q also accumulate gradients?
            # The reason for net, aux optimizers is just learning rate schedule...
            inputs["pdf_default"] = pdf_default

        if "cdf_default" in input_format:
            inputs["cdf_default"] = inputs["pdf_default"].cumsum(axis=-1)

        if any(k in input_format for k in ("pdf", "cdf")):
            y_actual = y_centered.reshape(*y_centered.shape[:2], -1)
            pdf_actual = self.histogram(y_actual, mass=True, hard=hard)
            inputs["pdf"] = pdf_actual

        if "cdf" in input_format:
            cdf_actual = pdf_actual.cumsum(axis=-1)
            inputs["cdf"] = cdf_actual

        pdf_input = torch.stack([inputs[k] for k in input_format], axis=1)

        return pdf_input, y

    def _compute_eb(self, pdf, q_indexes):
        _, num_bins = pdf.shape

        min_index = q_indexes[:, 0].clip(min=0)
        max_index = q_indexes[:, 2].clip(max=num_bins - 1)
        pmf_length = max_index - min_index + 1
        max_length = pmf_length.max().item()

        pmf = roll_along(pdf, -min_index, dim=-1)
        pmf = fill_runs(pmf, pmf_length - max_length, dim=-1, fill_value=0)
        tail_mass = (1 - pmf.sum(axis=-1)).clip(min=0).unsqueeze(-1)

        return {
            "_quantized_cdf": self.entropy_bottleneck._pmf_to_cdf(
                pmf, tail_mass, pmf_length, max_length
            ),
            "_cdf_length": pmf_length + 2,
            "_offset": min_index - q_indexes[:, 1],
        }

    @contextmanager
    def _override_eb(self, **kwargs):
        eb = self.entropy_bottleneck
        orig = {k: getattr(eb, k) for k in kwargs.keys()}
        for k, v in kwargs.items():
            setattr(eb, k, v)
        yield eb
        for k, v in orig.items():
            setattr(eb, k, v)

    def rdoq(self, y):
        y_offset = self.y_offset[None, :, None, None]
        y_hat = (y - y_offset).round() + y_offset
        return y_hat


class AdaptiveMixin(FreezeMixin):
    def __init__(
        self,
        enable_adaptive=True,
        num_bins=256,
        bound_factor=0.75,
        pdf_model_conf={
            "name": "um-pdf",
            "hp": {
                "C": 192,
                "N": 192,
                "M": 192,
                "input_format": ["pdf"],
                "act": "relu",
            },
        },
        freeze=(),
        unfreeze=(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pdf_model = MODELS[pdf_model_conf["name"]](**pdf_model_conf["hp"])

        self.latent_codec = nn.ModuleDict(
            {
                "pdf": AdaptiveLatentCodec(
                    entropy_bottleneck=self.entropy_bottleneck,
                    pdf_model=self.pdf_model,
                    enable_adaptive=enable_adaptive,
                    num_bins=num_bins,
                    bound_factor=bound_factor,
                )
            }
        )

        self._freeze_modules(freeze, unfreeze)


@register_model("bmshj2018-factorized-pdf")
class AdaptiveFactorizedPrior(FactorizedPrior, AdaptiveMixin):
    def __init__(self, N: int, M: int, **kwargs):
        FactorizedPrior.__init__(self, N=N, M=M)
        AdaptiveMixin.__init__(self, **kwargs)

    def forward(self, x):
        y = self.g_a(x)
        out_pdf = self.latent_codec["pdf"](y)
        y_hat = out_pdf["y_hat"]
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": out_pdf["likelihoods"],
            "p": out_pdf["p"],
            "p_hat": out_pdf["p_hat"],
        }

    @torch.no_grad()
    def compress(self, x):
        y = self.g_a(x)
        out_pdf = self.latent_codec["pdf"].compress(y)
        return {
            "strings": out_pdf["strings"],
            "shape": out_pdf["shape"],
            "pdf_x_default": out_pdf["pdf_x_default"],
            # Additional outputs:
            # "y_hat": y_hat,
            "p_hat": out_pdf["p_hat"],
        }

    @torch.no_grad()
    def decompress(self, strings, shape, pdf_x_default, **kwargs):
        assert isinstance(strings, list)
        out_pdf = self.latent_codec["pdf"].decompress(strings, shape, pdf_x_default)
        y_hat = out_pdf["y_hat"]
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


@register_model("bmshj2018-hyperprior-pdf")
class AdaptiveScaleHyperprior(ScaleHyperprior, AdaptiveMixin):
    def __init__(self, N: int, M: int, **kwargs):
        ScaleHyperprior.__init__(self, N=N, M=M)
        AdaptiveMixin.__init__(self, **kwargs)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        out_pdf = self.latent_codec["pdf"](z)
        z_hat = out_pdf["y_hat"]
        z_likelihoods = out_pdf["likelihoods"]["y"]

        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
                **{k: v for k, v in out_pdf["likelihoods"].items() if k not in ("y",)},
            },
            "p": out_pdf["p"],
            "p_hat": out_pdf["p_hat"],
        }

    @torch.no_grad()
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        # z_strings = self.entropy_bottleneck.compress(z)
        # z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        out_pdf = self.latent_codec["pdf"].compress(z)
        z_hat = out_pdf["y_hat"]
        z_stringses = out_pdf["strings"]

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return {
            "strings": [y_strings, *z_stringses],
            "shape": out_pdf["shape"],
            "pdf_x_default": out_pdf["pdf_x_default"],
            # Additional outputs:
            # "y_hat": y_hat,
            "p_hat": out_pdf["p_hat"],
        }

    @torch.no_grad()
    def decompress(self, strings, shape, pdf_x_default, **kwargs):
        assert isinstance(strings, list)

        # z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        out_pdf = self.latent_codec["pdf"].decompress(strings[1:], shape, pdf_x_default)
        z_hat = out_pdf["y_hat"]

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


@register_model("mbt2018-mean-pdf")
class AdaptiveMeanScaleHyperprior(MeanScaleHyperprior, AdaptiveMixin):
    def __init__(self, N: int, M: int, **kwargs):
        MeanScaleHyperprior.__init__(self, N=N, M=M)
        AdaptiveMixin.__init__(self, **kwargs)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        # z_hat, z_likelihoods = self.entropy_bottleneck(z)
        out_pdf = self.latent_codec["pdf"](z)
        z_hat = out_pdf["y_hat"]
        z_likelihoods = out_pdf["likelihoods"]["y"]

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
                **{k: v for k, v in out_pdf["likelihoods"].items() if k not in ("y",)},
            },
            "p": out_pdf["p"],
            "p_hat": out_pdf["p_hat"],
        }

    @torch.no_grad()
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        # z_strings = self.entropy_bottleneck.compress(z)
        # z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        out_pdf = self.latent_codec["pdf"].compress(z)
        z_hat = out_pdf["y_hat"]
        z_stringses = out_pdf["strings"]

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, *z_stringses],
            "shape": out_pdf["shape"],
            "pdf_x_default": out_pdf["pdf_x_default"],
            # Additional outputs:
            # "y_hat": y_hat,
            "p_hat": out_pdf["p_hat"],
        }

    @torch.no_grad()
    def decompress(self, strings, shape, pdf_x_default, **kwargs):
        assert isinstance(strings, list)

        # z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        out_pdf = self.latent_codec["pdf"].decompress(strings[1:], shape, pdf_x_default)
        z_hat = out_pdf["y_hat"]

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


def cross_entropy(p, q, eps=1e-12, axis=None):
    return -(p * (q + eps).log2()).sum(axis=axis)


def roll_along(arr, shifts, dim, fill_value=None):
    assert arr.ndim - 1 == shifts.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim], device=shifts.device).reshape(shape)
    shifts_ = shifts.unsqueeze(dim)

    if fill_value is not None:
        runs_ = -shifts_
        left_mask = (runs_ >= 0) & (dim_indices >= runs_)
        right_mask = (runs_ < 0) & (dim_indices < runs_ + arr.shape[dim])
        mask = left_mask | right_mask
        arr = arr.where(mask, torch.full_like(arr, fill_value))

    indices = (dim_indices - shifts_) % arr.shape[dim]
    arr = torch.gather(arr, dim, indices)
    return arr


def fill_runs(arr, runs, dim, fill_value=0):
    assert arr.ndim - 1 == runs.ndim
    dim %= arr.ndim
    shape = (1,) * dim + (-1,) + (1,) * (arr.ndim - dim - 1)
    dim_indices = torch.arange(arr.shape[dim], device=runs.device).reshape(shape)
    runs_ = runs.unsqueeze(dim)
    left_mask = (runs_ >= 0) & (dim_indices >= runs_)
    right_mask = (runs_ < 0) & (dim_indices < runs_ + arr.shape[dim])
    mask = left_mask | right_mask
    arr = arr.where(mask, torch.full_like(arr, fill_value))
    return arr


# NOTE Careful: some are pdf, some are cdf... e.g. histogram assumes pdf?

# TODO Improve RD performance via:
#
# - Bound quantiles
# - CDF logits or CDF or "PDF logits" instead of PDF?
# - Sample rate = 4
# - Discrete indexing loss
# - Ensure gradients flow through PDF model and g_a
# - Multi-stage training (i.e. train pdf_model on frozen pretrained model; then joint)
# - Repair y bounds
# - Repair y shifted

# TODO Improve training performance via:
#
# - ...

# TODO Merge with RDOQ.

# TODO Other interesting ideas:
#
# - GaussianConditional + PDF model
# - Entropy bottleneck "matrix" as a dynamic function of the PDF model

# Override entropy bottleneck's forward pass to use the pdf model's...
# How to deal with batches? Need to override n-D weights with (n+1)-D weights?!
#
# NOTE: This is over a batch... so can't just do one!
# Might need to rewrite our own entropy_bottleneck, or hack into
# what differentiable PDF method it uses?
# Maybe just skip, and write the compression/decompression methods instead?

# Interleaved/"auxillary" training...? Don't really need the aux framework.
# Can even initialize to null-string like in V-information or something.

# Deal with outside symbol_boundaries, too...?

# Create an r-compatible PdfModel

# Blend with pdf_default
# r = pdf_hat[..., -1, None]
# pdf_hat_fullsize = pdf_hat_fullsize + r * pdf_default_fullsize

# NOTE: Due to the DOF constraint on PDFs (sum to 1),
# we don't actually need to run PdfModel on special symbols...!
# Just need to use an RD loss with the extra pseudo-symbol:
# r = (1 - pdf.sum()).clip(min=0)
#
# NOTE: Without taking into account tail_mass,
# the PDF compression model isn't completely "accurate".
#
# Still need to deal with out-of-bounds...?
# Or instead, rely upon conv behavior...? Does conv behavior generalize?
