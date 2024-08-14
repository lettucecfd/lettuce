from abc import ABC
from typing import List

import torch
import numpy as np

from ._context import Context

__all__ = ['Stencil', 'TorchStencil']


class Stencil(ABC):
    e: List[List[int]]
    w: List[float]
    opposite: List[int]

    cs: float = 1 / np.sqrt(3.0)

    @property
    def d(self):
        return len(self.e[0])

    @property
    def q(self):
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

    @property
    def d(self):
        return self.e.shape[1]

    @property
    def q(self):
        return self.e.shape[0]
