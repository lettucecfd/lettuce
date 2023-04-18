"""
Streaming Step
"""

import torch
import numpy as np

from typing import Optional
from lettuce.base import LatticeBase
from lettuce.native_generator import NativeNoStreaming, NativeStandardStreaming

__all__ = ["StandardStreaming", "NoStreaming"]


class Streaming(LatticeBase):
    no_streaming_mask: Optional[torch.Tensor]

    def __init__(self, lattice: 'Lattice'):
        LatticeBase.__init__(self, lattice)
        self.no_streaming_mask = None

    def __call__(self, f):
        raise NotImplementedError()

    # attributes for backwards compatibility

    @property
    def no_stream_mask(self):
        return self.no_streaming_mask

    @no_stream_mask.setter
    def no_stream_mask(self, mask):
        self.no_streaming_mask = mask


class NoStreaming(Streaming):
    def __init__(self, lattice: 'Lattice'):
        Streaming.__init__(self, lattice)

    def native_available(self) -> bool:
        return True

    def create_native(self) -> 'NativeNoStreaming':
        return NativeNoStreaming()

    def __call__(self, f):
        return f


class StandardStreaming(Streaming):
    """Standard Streaming step on a regular grid."""

    def native_available(self) -> bool:
        return True

    def create_native(self) -> 'NativeStandardStreaming':
        support_no_streaming_mask = (self.no_streaming_mask is not None) and self.no_streaming_mask.any()
        return NativeStandardStreaming(support_no_streaming_mask)

    def __call__(self, f):
        for i in range(1, self.lattice.Q):
            if self.no_stream_mask is None:
                f[i] = self._stream(f, i)
            else:
                new_fi = self._stream(f, i)
                f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)
        return f

    def _stream(self, f, i):
        return torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))


class SLStreaming(Streaming):
    """
    TODO (is there a good python package for octrees or do we have to write this ourselves?)
    """

    def __init__(self, lattice: 'Lattice', grid):
        Streaming.__init__(self, lattice)
        raise NotImplementedError()
