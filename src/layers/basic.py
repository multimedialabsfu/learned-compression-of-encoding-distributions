import torch.nn as nn

from torch import Tensor


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class Interleave(nn.Module):
    def __init__(self, groups: int = 1):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        n, c, *tail = x.shape
        return (
            x.reshape(n, self.groups, c // self.groups, *tail)
            .transpose(1, 2)
            .reshape(x.shape)
        )
