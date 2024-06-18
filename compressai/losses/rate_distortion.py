from collections import defaultdict

import torch.nn as nn
from pytorch_msssim import ms_ssim

from compressai.registry import register_criterion


@register_criterion("AdaptiveRateDistortionLoss")
class AdaptiveRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(
        self,
        lmbda=0.01,
        metric="mse",
        return_type="all",
        target_shape=(512, 768),
    ):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type
        self.num_pixels_target = target_shape[0] * target_shape[1]

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        lmbda = defaultdict(lambda: 1.0)
        lmbda["pdf"] = (num_pixels / N) / self.num_pixels_target

        group_keys = set(key.split("_")[0] for key in output["likelihoods"].keys())

        for group_key in group_keys:
            out[f"bpp_{group_key}_loss"] = sum(
                likelihoods.log2().sum() / -num_pixels
                for key, likelihoods in output["likelihoods"].items()
                if key == group_key or key.startswith(f"{group_key}_")
            )

        out["bpp_loss"] = sum(
            lmbda[group_key] * out[f"bpp_{group_key}_loss"] for group_key in group_keys
        )

        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
