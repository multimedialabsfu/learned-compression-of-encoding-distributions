import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import LatentCodec
from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec
from compressai.layers.basic import Interleave, Reshape
from compressai.registry import register_model

from .base import CompressionModel


class PdfCompressionModel(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── x_hat
               └───┘     └────┘       └───┘
    """

    g_a: nn.Module
    g_s: nn.Module
    latent_codec: LatentCodec

    def __init__(self, input_format, ideal=False, **kwargs):
        super().__init__(**kwargs)

        self.input_format = input_format
        self._default_idxs = [i for i, s in enumerate(input_format) if "default" in s]
        self.ideal = ideal  # Ideal codec has 0 bit cost and 0 distortion.

    def forward(self, x):
        x_default = x[:, self._default_idxs]
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        y_hat_a = y_out["y_hat"]
        if self.g_a_b is not None:
            y_hat_b = self.g_a_b(x_default)  # No need to transmit default.
        else:
            y_hat_b = y_hat_a.new_zeros((y_hat_a.shape[0], 0, *y_hat_a.shape[2:]))
        y_hat = torch.cat([y_hat_a, y_hat_b], dim=1)
        x_hat = self.g_s(y_hat)
        if self.ideal:
            y_out["likelihoods"] = {
                k: torch.ones_like(v) for k, v in y_out["likelihoods"].items()
            }
            x_hat = x[:, 0]  # NOTE: Assumes 0th input is the desired target.
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x):
        x_default = x[:, self._default_idxs]
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)
        if self.ideal:
            outputs["strings"] = [[b"" for _ in ss] for ss in outputs["strings"]]
            x_default = x[:, 0]  # NOTE: Assumes 0th input is the desired target.
        return {**outputs, "x_default": x_default}

    def decompress(self, strings, shape, x_default, **kwargs):
        if self.ideal:
            x_hat = x_default
            return {"x_hat": x_hat}
        y_out = self.latent_codec.decompress(strings, shape)
        y_hat_a = y_out["y_hat"]
        if self.g_a_b is not None:
            y_hat_b = self.g_a_b(x_default)  # Decoder has access to default.
        else:
            y_hat_b = y_hat_a.new_zeros((y_hat_a.shape[0], 0, *y_hat_a.shape[2:]))
        y_hat = torch.cat([y_hat_a, y_hat_b], dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
        }


@register_model("um-pdf")
class PdfModel(PdfCompressionModel):
    def __init__(
        self,
        C,
        N,
        M,
        K=15,
        input_format=("pdf",),
        act="sigmoid",
        groups=8,
        **kwargs,
    ):
        super().__init__(input_format=input_format, **kwargs)

        # TODO "basis"/"wavelet"/"GMM" functions, multiplication, attention
        # TODO monotonic functions?

        F = len(self.input_format)
        D = len([s for s in self.input_format if "default" in s])

        make_act = {
            "sigmoid": nn.Sigmoid,
            "relu": lambda: nn.ReLU(inplace=True),
            "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        }[act]

        self.g_a = nn.Sequential(
            Reshape((F * C, -1)),
            conv1d(F * C, max(N, C // 2), kernel_size=5, groups=groups),
            Interleave(groups),
            make_act(),
            conv1d(max(N, C // 2), N, kernel_size=K, stride=2, groups=groups),
            Interleave(groups),
            make_act(),
            conv1d(N, N, kernel_size=K, stride=2, groups=groups),
            Interleave(groups),
            make_act(),
            conv1d(N, N, kernel_size=K, stride=2, groups=groups),
            make_act(),
            conv1d(N, M, kernel_size=K),
        )

        if D != 0:
            self.g_a_b = nn.Sequential(
                Reshape((D * C, -1)),
                conv1d(D * C, max(N, C // 2), kernel_size=5, groups=groups),
                Interleave(groups),
                make_act(),
                conv1d(max(N, C // 2), N, kernel_size=K, stride=2, groups=groups),
                Interleave(groups),
                make_act(),
                conv1d(N, N, kernel_size=K, stride=2, groups=groups),
                Interleave(groups),
                make_act(),
                conv1d(N, N, kernel_size=K, stride=2, groups=groups),
                make_act(),
                conv1d(N, M, kernel_size=K),
            )
        else:
            self.g_a_b = None

        self.g_s = nn.Sequential(
            deconv1d(M + M * (self.g_a_b is not None), N, kernel_size=K),
            make_act(),
            deconv1d(N, N, kernel_size=K, stride=2, groups=groups),
            Interleave(groups),
            make_act(),
            deconv1d(N, N, kernel_size=K, stride=2, groups=groups),
            Interleave(groups),
            make_act(),
            deconv1d(N, max(N, C // 2), kernel_size=K, stride=2, groups=groups),
            Interleave(groups),
            make_act(),
            deconv1d(max(N, C // 2), C, kernel_size=5, groups=groups),
            # make_act(),
            # nn.LogSoftmax(dim=-1),
            nn.Softmax(dim=-1),  # Enforce [0, 1] range.
        )

        self.latent_codec = EntropyBottleneckLatentCodec(
            entropy_bottleneck=EntropyBottleneck(M),
            dims=(-1,),
        )


def conv1d(in_channels, out_channels, kernel_size=1, stride=1, **kwargs):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        **kwargs,
    )


def deconv1d(in_channels, out_channels, kernel_size=1, stride=1, **kwargs):
    return nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        **kwargs,
    )
