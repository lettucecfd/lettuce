"""
Streaming Step
"""
from typing import Optional

import numpy as np
import torch

from . import *
from .native_generator import NativeNoStreaming, NativeStandardStreaming


class Streaming(LatticeBase):
    no_streaming_mask: Optional[torch.Tensor]

    def __init__(self, lattice: Lattice, use_native=True):
        super().__init__(lattice, use_native)

    def __call__(self, f):
        raise AbstractMethodInvokedError()


class NoStreaming(Streaming):
    def __init__(self, lattice: Lattice, use_native=True):
        super().__init__(lattice, use_native)
        self.no_streaming_mask = None

    def native_available(self) -> bool:
        return True

    def create_native(self) -> NativeNoStreaming:
        return NativeNoStreaming()

    def __call__(self, f):
        return f


class StandardStreaming(Streaming):
    """Standard Streaming step on a regular grid.

    Attributes
    ----------
    no_streaming_mask : torch.Tensor
        Boolean mask with the same shape as the distribution function f.
        If None, stream all (also around all boundaries).
    """

    native_class = native_generator.NativeStandardStreaming

    def __init__(self, lattice: Lattice, use_native=True):
        super().__init__(lattice, use_native)
        self.no_streaming_mask = None

    def native_available(self) -> bool:
        return True

    def create_native(self) -> NativeStandardStreaming:
        return NativeStandardStreaming(self.no_streaming_mask is not None)

    def _stream(self, f, i):
        return torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.d)))

    def __call__(self, f):
        for i in range(1, self.lattice.q):
            if self.no_streaming_mask is None:
                f[i] = self._stream(f, i)
            else:
                new_fi = self._stream(f, i)
                f[i] = torch.where(self.no_streaming_mask[i], f[i], new_fi)
        return f


class SLStreaming(Streaming):
    """
    TODO (is there a good python package for octrees or do we have to write this ourselves?)
    """

    def __init__(self, lattice: Lattice, grid, use_native=True):
        super().__init__(lattice, use_native)
        raise NotImplementedError
