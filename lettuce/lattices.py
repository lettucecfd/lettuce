"""
Stencils and Lattices.

A Stencil, like the D1Q3 class, provides particle velocities (e), weights (w), and speeds of sound.
Velocities and weights are stored as numpy arrays.

In contrast, the Lattice lives on the Device (usually a GPU), and its
"""

import warnings
import numpy as np
import torch


class Stencil(object):
    pass


class D1Q3(Stencil):
    e = np.array([[0], [1], [-1]])
    w = np.array([2.0/3.0, 1.0/6.0, 1.0/6.0])
    cs = 1/np.sqrt(3)


class D2Q9(Stencil):
    e = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
    w = np.array([4.0/9.0] + [1.0/9.0]*4 + [1.0/36.0]*4)
    cs = 1/np.sqrt(3)


class Lattice(object):
    def __init__(self, lattice, device, dtype=torch.float):
        self.stencil = lattice
        self.device = device
        self.dtype = dtype
        self.e = self.convert_to_tensor(lattice.e)
        self.w = self.convert_to_tensor(lattice.w)
        self.cs = self.convert_to_tensor(lattice.cs)

    def __str__(self):
        return f"Lattice (stencil {self.stencil.__name__}; device {self.device}; dtype {self.dtype})"

    @property
    def D(self):
        return self.e.shape[1]

    @property
    def Q(self):
        return self.e.shape[0]

    def convert_to_tensor(self, array):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.tensor(array, device=self.device, dtype=self.dtype)

    def rho(self, f):
        """density"""
        return torch.sum(f, dim=0)

    def j(self, f):
        """momentum"""
        return torch.einsum("qd,q...->d...", [self.e, f])

    def u(self, f):
        """velocity"""
        return self.j(f) / self.rho(f)

    def quadratic_equilibrium(self, rho, u):
        exu = torch.einsum("qd,d...->q...", [self.e, u])
        uxu = torch.einsum("d...,d...->...", [u,u])
        feq = torch.einsum(
            "q,q...->q...",
            [self.w, rho * ((2 * exu - uxu) / (2 * self.cs ** 2) + 0.5 * (exu / (self.cs ** 2)) ** 2 + 1)])
        return feq
