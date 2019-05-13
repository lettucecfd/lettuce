"""
Streaming Step
"""

import torch
import numpy as np
from lettuce import LatticeOfVector, Lattice


class StandardStreaming(object):
    """Standard Streaming step on a regular grid."""
    def __init__(self, lattice):
        self.lattice = lattice

    def __call__(self, f):
        for i in range(self.lattice.Q):
            if isinstance(self.lattice, LatticeOfVector):
                f[...,i] = torch.roll(f[...,i],
                                      shifts=tuple(self.lattice.stencil.e[i]),
                                      dims=tuple(np.arange(self.lattice.D)))
            else:
                f[i] = torch.roll(f[i],
                                  shifts=tuple(self.lattice.stencil.e[i]),
                                  dims=tuple(np.arange(self.lattice.D)))
        return f


class SLStreaming(object):
    """
    TODO (is there a good python package for octrees or do we have to write this ourselves?)
    """
    def __init__(self, lattice, grid):
        raise NotImplementedError
