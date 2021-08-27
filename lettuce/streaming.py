"""
Streaming Step
"""

import numpy as np
import torch

from lettuce import *

__all__ = ["NoStreaming", "StandardStreaming"]


class NoStreaming:
    """
    """

    native_class = gen_native.NativeStreamingNo

    def __init__(self):
        self._no_stream_mask = None

    @property
    def no_stream_mask(self):
        return self._no_stream_mask

    @no_stream_mask.setter
    def no_stream_mask(self, mask):
        self._no_stream_mask = mask

    def __call__(self, f):
        return f


class StandardStreaming:
    """Standard Streaming step on a regular grid.

    Attributes
    ----------
    no_stream_mask : torch.Tensor
        Boolean mask with the same shape as the distribution function f.
        If None, stream all (also around all boundaries).
    """

    native_class = gen_native.NativeStreamingStandard

    def __init__(self, lattice):
        self.lattice = lattice
        self._no_stream_mask = None

    @property
    def no_stream_mask(self):
        return self._no_stream_mask

    @no_stream_mask.setter
    def no_stream_mask(self, mask):
        self._no_stream_mask = mask

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


class SLStreaming:
    """
    TODO (is there a good python package for octrees or do we have to write this ourselves?)
    """

    def __init__(self, lattice, grid):
        raise NotImplementedError
