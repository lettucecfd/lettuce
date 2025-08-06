from abc import ABC
from typing import cast
from functools import cached_property
from typing import List, Literal

import torch
import numpy as np

from ._context import Context

__all__ = ['Stencil', 'TorchStencil']


class Stencil(ABC):
    e: List[List[int]]
    w: List[float]
    opposite: List[int]

    cs: float = 1 / np.sqrt(3.0)

    @cached_property
    def d(self) -> int:
        assert len(self.e[0]) in [1, 2, 3]
        return len(self.e[0])

    @cached_property
    def q(self) -> int:
        return len(self.e)


class TorchStencil:
    e: torch.Tensor
    w: torch.Tensor
    opposite: torch.Tensor
    cs: float = 1 / np.sqrt(3.0)

    def __init__(self, stencil: 'Stencil', context: 'Context'):
        self.e = context.convert_to_tensor(stencil.e)
        self.w = context.convert_to_tensor(stencil.w)
        self.opposite = context.convert_to_tensor(stencil.opposite)
        self.e = context.convert_to_tensor(stencil.e)

    @cached_property
    def d(self) -> Literal[1, 2, 3]:
        assert self.e.shape[1] in [1, 2, 3]
        return cast(Literal[1, 2, 3], self.e.shape[1])

    @cached_property
    def q(self) -> int:
        return self.e.shape[0]
