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

from lettuce.util import LettuceException
from lettuce.equilibrium import QuadraticEquilibrium


class Lattice:

    def __init__(self, stencil, device, dtype=torch.float):
        self.stencil = stencil
        self.device = device
        self.dtype = dtype
        self.e = self.convert_to_tensor(stencil.e)
        self.w = self.convert_to_tensor(stencil.w)
        self.cs = self.convert_to_tensor(stencil.cs)
        self.equilibrium = QuadraticEquilibrium(self)

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

    @classmethod
    def convert_to_numpy(cls, tensor):
        return tensor.detach().cpu().numpy()

    def rho(self, f):
        """density"""
        return torch.sum(f, dim=0)[None,...]

    def j(self, f):
        """momentum"""
        return self.einsum("qd,q->d", [self.e, f])

    def u(self, f):
        """velocity"""
        return self.j(f) / self.rho(f)

    def incompressible_energy(self, f):
        """incompressible kinetic energy"""
        return 0.5*self.einsum("d,d->", [self.u(f), self.u(f)])

    def entropy(self, f):
        """entropy according to the H-theorem"""
        f_log = -torch.log(self.einsum("q,q->q",[f,1/self.w]))
        return self.einsum("q,q->", [f,f_log])

    def pseudo_entropy_global(self,f):
        """pseudo_entropy derived by a Taylor expansion around the weights"""
        f_w = self.einsum("q,q->q", [f, 1 / self.w])
        return self.rho(f) - self.einsum("q,q->", [f,f_w])

    def pseudo_entropy_local(self,f):
        """pseudo_entropy derived by a Taylor expansion around the local equilibrium"""
        f_feq = f/self.equilibrium(self.rho(f),self.u(f))
        return self.rho(f) - self.einsum("q,q->", [f,f_feq])

    def shear_tensor(self, f):
        """computes the shear tensor of a given f in the sense Pi_{\alpha \beta} = f_i * e_{i \alpha} * e_{i \beta}"""
        shear = self.einsum("qa,qb->qab", [self.e, self.e])
        shear = self.einsum("q,qab->ab", [f, shear])
        return shear

    def mv(self, m, v):
        """matrix-vector multiplication"""
        return self.einsum("ij,j->i", [m,v])

    def einsum(self, equation, fields, **kwargs):
        """Einstein summation on local fields."""
        input, output = equation.split("->")
        inputs = input.split(",")
        for i,inp in enumerate(inputs):
            if len(inp) == len(fields[i].shape):
                pass
            elif len(inp) == len(fields[i].shape) - self.D:
                inputs[i] += "..."
                if not output.endswith("..."):
                    output += "..."
            else:
                raise LettuceException("Bad dimension.")
        equation = ",".join(inputs) + "->" + output
        return torch.einsum(equation, fields, **kwargs)

    def torch_gradient(self, f, dx=1):
        dim = f.ndim
        if dim==2:
            dims = (0, 1)
            shift = [[[-2, 0], [-1, 0], [1, 0], [2, 0]],
                     [[0, -2], [0, -1], [0, 1], [0, 2]]]
        if dim==3:
            dims = (0,1,2)
            ## Stencil for 2nd order
            # shift = [[[-1, 0, 0], [1, 0, 0]],
            #          [[0, -1, 0], [0, 1, 0]],
            #          [[0, 0, -1], [0, 0, 1]]]
            ## Stencil for 4th order
            # shift = [[[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0]],
            #          [[0, -2, 0], [0, -1, 0], [0, 1, 0], [0, 2, 0]],
            #          [[0, 0, -2], [0, 0, -1], [0, 0, 1], [0, 0, -2]]]
            ## Stencil for 6th order
            shift = [[[-3, 0, 0], [-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                     [[0, -3, 0], [0, -2, 0], [0, -1, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]],
                     [[0, 0, -3], [0, 0, -2], [0, 0, -1], [0, 0, 1], [0, 0, 2], [0, 0, 3]]]
        with torch.no_grad():
            out = torch.cat(dim*[f[None,...]])
            # for i in range(dim):
            #     out[i, ...] = (1/12 * f.roll(shifts=shift[i][0], dims=dims) +
            #                    -2/3 * f.roll(shifts=shift[i][1], dims=dims) +
            #                    2/3 * f.roll(shifts=shift[i][2], dims=dims) +
            #                    -1/12 * f.roll(shifts=shift[i][3], dims=dims)) / (dx)
            # for i in range(dim):
            #     out[i, ...] = (-1/2 * f.roll(shifts=shift[i][0], dims=dims) +
            #                    1/2 * f.roll(shifts=shift[i][1], dims=dims)) / (dx)
            for i in range(dim):
                out[i, ...] = (-1/60 * f.roll(shifts=shift[i][0], dims=dims) +
                               3/20 * f.roll(shifts=shift[i][1], dims=dims) +
                               -3/4 * f.roll(shifts=shift[i][2], dims=dims) +
                               3/4 * f.roll(shifts=shift[i][3], dims=dims) +
                               -3/20 * f.roll(shifts=shift[i][4], dims=dims) +
                               1/60 * f.roll(shifts=shift[i][5], dims=dims)) / (dx)
        return out


