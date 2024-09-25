import torch

from typing import List, Optional
from ... import Boundary, Flow, Context
from ...cuda_native.ext import NativeBounceBackBoundary

__all__ = ['BounceBackBoundary']


class BounceBackBoundary(Boundary):
    _mask: torch.Tensor

    def __init__(self, mask: torch.Tensor):
        self._mask = mask
        return

    def __call__(self, flow: 'Flow'):
        return flow.f[flow.stencil.opposite]

    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        return None

    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        return self._mask

    def native_available(self) -> bool:
        return True

    def native_generator(self, index: int) -> 'NativeBoundary':
        return NativeBounceBackBoundary(index)
