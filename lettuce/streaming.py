"""
Streaming Step
"""

import torch
import numpy as np


class StandardStreaming:
    """Standard Streaming step on a regular grid."""
    def __init__(self, lattice):
        self.lattice = lattice

    def __call__(self, f):
        for i in range(self.lattice.Q):
            f[i] = torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))
        return f

class IO_2D_Streaming:
    """Special Streaming step on a regular grid."""
    def __init__(self, lattice, grid):
        self.lattice = lattice
        self.save_f = lattice.convert_to_tensor(np.zeros([3, 150]))

    def __call__(self, f):
        count = 0
        for i in [3, 7, 6]:
            self.save_f[count] = f[i, -1, :]
            count += 1
        for i in range(self.lattice.Q):
            f[i] = torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))
        count = 0
        for i in [3, 7, 6]:
            f[i, -1, :] = self.save_f[count]
            count += 1
        return f


class SLStreaming:
    """
    TODO (is there a good python package for octrees or do we have to write this ourselves?)
    """
    def __init__(self, lattice, grid):
        raise NotImplementedError
