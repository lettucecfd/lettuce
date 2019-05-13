"""
Stencils and Lattices.

A Stencil, like the D1Q3 class, provides particle velocities (e), weights (w), and speeds of sound.
Velocities and weights are stored as numpy arrays.

In contrast, the Lattice lives on the Device (usually a GPU) and its vectors are stored as torch tensors.
Its stencil is still accessible trough Lattice.stencil.
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
    opposite = [0, 2, 1]


class D2Q9(Stencil):
    e = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
    w = np.array([4.0/9.0] + [1.0/9.0]*4 + [1.0/36.0]*4)
    cs = 1/np.sqrt(3)
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]


class Lattice(object):
    def __init__(self, stencil, device, dtype=torch.float):
        self.stencil = stencil
        self.device = device
        self.dtype = dtype
        self.e = self.convert_to_tensor(stencil.e)
        self.w = self.convert_to_tensor(stencil.w)
        self.cs = self.convert_to_tensor(stencil.cs)

    def __str__(self):
        return f"Lattice (stencil {self.stencil.__name__}; device {self.device}; dtype {self.dtype})"

    @property
    def D(self):
        return self.stencil.e.shape[1]

    @property
    def Q(self):
        return self.stencil.e.shape[0]

    def convert_to_tensor(self, array):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(array, np.ndarray) and array.dtype in [np.bool, np.uint8]:
                return torch.tensor(array, device=self.device, dtype=torch.uint8)  # that's how torch stores its masks
            else:
                return torch.tensor(array, device=self.device, dtype=self.dtype)

    @staticmethod
    def convert_to_numpy(tensor):
        return tensor.cpu().numpy()

    def rho(self, f):
        """density"""
        return torch.sum(f, dim=0)[None, :]

    def j(self, f):
        """momentum"""
        j = torch.einsum("qd,q...->d...", [self.e, f])
        if self.D == 1:
            return j
        else:
            return j

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


class LatticeOfVector(Lattice):
    """
    Lattice class with inverse storage order
    """
    def __init__(self, stencil, device, dtype=torch.float):
        super(LatticeOfVector,self).__init__(stencil, device, dtype)
        self.e = self.convert_to_tensor(stencil.e)
        self.w = self.convert_to_tensor(stencil.w)

    def convert_to_tensor(self, array):
        try:
            a = np.moveaxis(array, 0, -1)
        except: a = array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(a, np.ndarray) and a.dtype in [np.bool, np.uint8]:
                return torch.tensor(a, device=self.device, dtype=torch.uint8)  # that's how torch stores its masks
            else:
                return torch.tensor(a, device=self.device, dtype=self.dtype)

    def __str__(self):
        return f"LatticeOfArray (stencil {self.stencil.__name__}; device {self.device}; dtype {self.dtype})"

    @staticmethod
    def convert_to_numpy(tensor):
        return np.moveaxis(tensor.cpu().numpy(), -1, 0)

    @staticmethod
    def add_last_dimension(tensor):
        return tensor.view(list(tensor.shape) + [1])

    def rho(self, f):
        """density"""
        return self.check_rho_dim(self.add_last_dimension(torch.sum(f, dim=-1)))

    def j(self, f):
        """momentum"""
        return self.check_j_dim(torch.einsum("dq,...q->...d", [self.e, f]))

    def u(self, f):
        """velocity"""
        return self.check_j_dim(self.j(f) / self.rho(f))

    def quadratic_equilibrium(self, rho, u):
        exu = self.check_dim(torch.einsum("dq,...d->...q", [self.e, u]), length=self.D+1, last=self.Q)
        uxu = self.check_dim(self.add_last_dimension(torch.einsum("...d,...d->...", [u,u])), length=self.D+1, last=1)
        feq = torch.einsum(
            "q,...q->...q",
            [self.w, rho * ((2 * exu - uxu) / (2 * self.cs ** 2) + 0.5 * (exu / (self.cs ** 2)) ** 2 + 1)])
        return feq

    def check_rho_dim(self, rho):
        assert len(rho.size()) == self.D + 1
        assert rho.size()[-1] == 1
        return rho

    def check_j_dim(self, j):
        assert len(j.size()) == self.D + 1
        assert j.size()[-1] == self.D
        return j

    @staticmethod
    def check_dim(tensor, length=None, first=None, last=None):
        size = tensor.size()
        if length is not None: assert len(size) == length
        if first is not None: assert size[0] == first
        if last is not None: assert size[-1] == last
        return tensor

