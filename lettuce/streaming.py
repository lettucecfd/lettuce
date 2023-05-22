"""
Streaming Step
"""

import torch
import numpy as np

from typing import Optional
from lettuce.base import LettuceBase
from lettuce.native_generator import NativeLettuceBase, NativeWrite, NativeRead, NativeStandardStreamingWrite, NativeStandardStreamingRead

__all__ = ["StandardStreaming", "NoStreaming"]


class Streaming(LettuceBase):
    no_streaming_mask: Optional[torch.Tensor]

    def __init__(self, lattice: 'Lattice'):
        LettuceBase.__init__(self, lattice)
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
    def __call__(self, f):
        return f

    def native_available(self) -> bool:
        return True

    def create_native(self) -> ['NativeLettuceBase']:
        return [NativeRead(), NativeWrite()]


class Read(NoStreaming):
    def create_native(self) -> ['NativeLettuceBase']:
        return [NativeRead()]


class Write(NoStreaming):
    def create_native(self) -> ['NativeLettuceBase']:
        return [NativeWrite()]


class StandardStreaming(Streaming):
    """Standard Streaming step on a regular grid."""

    def native_available(self) -> bool:
        return True

    def create_native(self) -> ['NativeLettuceBase']:
        support_no_streaming_mask = (self.no_streaming_mask is not None) and self.no_streaming_mask.any()
        return [NativeRead(), NativeStandardStreamingWrite(support_no_streaming_mask)]
        # return [NativeStandardStreamingRead(support_no_streaming_mask), NativeWrite()]

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


class StandardRead(StandardStreaming, Read):
    def create_native(self) -> ['NativeLettuceBase']:
        support_no_streaming_mask = (self.no_streaming_mask is not None) and self.no_streaming_mask.any()
        return [NativeStandardStreamingRead(support_no_streaming_mask), NativeWrite()]


class StandardWrite(StandardStreaming, Write):
    def create_native(self) -> ['NativeLettuceBase']:
        support_no_streaming_mask = (self.no_streaming_mask is not None) and self.no_streaming_mask.any()
        return [NativeRead(), NativeStandardStreamingWrite(support_no_streaming_mask)]


class SLStreaming(Streaming):
    """
    TODO (is there a good python package for octrees or do we have to write this ourselves?)
    """

    def __call__(self, f):
        raise NotImplementedError()

    def native_available(self) -> bool:
        return False

    def __init__(self, lattice: 'Lattice', *_):
        Streaming.__init__(self, lattice)
        raise NotImplementedError()

    def create_native(self) -> ['NativeLettuceBase']:
        raise NotImplementedError()
