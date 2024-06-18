# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

from functools import reduce
from typing import Optional

import torch
import torch.nn as nn

from torch import Tensor


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


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
