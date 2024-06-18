import json

import numpy as np

from torch.utils.data import Dataset

from compressai.ops.ops import compute_padding_1d
from compressai.registry import register_dataset


@register_dataset("PdfDataset")
class PdfDataset(Dataset):
    def __init__(
        self,
        path,
        path_default=None,
        format=("pdf",),
        min_div=64,
        transform=None,
    ):
        self.path = path
        self.path_default = path_default
        self.format = format
        self.min_div = min_div
        assert not transform or len(transform.transforms) == 0
        self.transform = transform
        with open(self.path.rstrip(".npy") + ".meta.json", "r") as f:
            self.meta = json.load(f)
        self.items = np.memmap(
            self.path,
            mode="r",
            dtype=self.meta["dtype"],
            shape=tuple(self.meta["shape"]),
        )
        self.pdf_default = (
            np.memmap(
                self.path_default,
                mode="r",
                dtype="float64",
                shape=tuple(self.meta["shape"])[1:],
            )
            if self.path_default is not None
            else None
        )
        self.eps = 1e-12
        self._pad, self._unpad = self._compute_padding()
        self.pdf_default_padded = np.pad(self.pdf_default.astype(np.float32), self._pad)

    def __getitem__(self, index):
        pdf = self.items[index] / self.meta["max_pdf_value"]
        pdf = np.pad(pdf.astype(np.float32), self._pad)
        pdf_default = self.pdf_default_padded

        output = []

        for fmt in self.format:
            if fmt == "pdf":
                output.append(pdf)
            if fmt == "cdf":
                cdf = pdf.cumsum(axis=-1)
                output.append(cdf)
            if fmt == "log_pdf":
                log_pdf = np.log2(pdf + self.eps)
                output.append(log_pdf)
            if fmt == "pdf_default":
                output.append(pdf_default)
            if fmt == "cdf_default":
                cdf_default = pdf_default.cumsum(axis=-1)
                output.append(cdf_default)
            if fmt == "log_pdf_default":
                log_pdf_default = np.log2(pdf_default + self.eps)
                output.append(log_pdf_default)

        output = np.stack(output)
        return output

    def __len__(self):
        return self.meta["shape"][0]

    def _compute_padding(self):
        _, _, P = self.meta["shape"]
        pad, unpad = compute_padding_1d(P, min_div=self.min_div)
        pad = ((0, 0), pad)
        unpad = ((0, 0), unpad)
        return pad, unpad
