import math
from functools import reduce
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RangeBoundFunction(torch.autograd.Function):
    """Autograd function for the ``RangeBound`` operator."""

    @staticmethod
    def forward(ctx, x, bound_min=None, bound_max=None):
        ctx.save_for_backward(x, bound_min, bound_max)
        return torch.clip(x, min=bound_min, max=bound_max)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound_min, bound_max = ctx.saved_tensors
        violates = []
        if bound_min is not None:
            violates.append((x < bound_min) & (grad_output >= 0))
        if bound_max is not None:
            violates.append((x > bound_max) & (grad_output <= 0))
        if len(violates) == 0:
            return grad_output, None, None
        pass_through_if = ~(reduce(lambda x, y: x | y, violates))
        return pass_through_if * grad_output, None, None


class RangeBound(nn.Module):
    """Range bound operator, computes
    ``torch.clip(x, bound_min, bound_max)`` with a custom gradient.

    Within the range, the gradient is passed through unchanged.
    Outside the range, only gradients that direct the input ``x`` to
    move back inside the range are kept; otherwise, the gradients are
    set to zero.
    """

    bound: Tensor

    def __init__(self, min: Optional[float] = None, max: Optional[float] = None):
        super().__init__()
        bound = [
            float("nan" if min is None else min),
            float("nan" if max is None else max),
        ]
        self.register_buffer("bound", torch.Tensor(bound))

    @property
    def bound_min(self):
        x = self.bound[0]
        return None if math.isnan(x.item()) else x

    @property
    def bound_max(self):
        x = self.bound[1]
        return None if math.isnan(x.item()) else x

    @torch.jit.unused
    def range_bound(self, x):
        return RangeBoundFunction.apply(x, self.bound_min, self.bound_max)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.clip(x, min=self.bound_min, max=self.bound_max)
        return self.range_bound(x)
