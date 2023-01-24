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

from typing import Type

from lettuce.util import LettuceException
from lettuce.equilibrium import QuadraticEquilibrium

__all__ = ["Lattice"]


class Lattice:

    stencil: Type['Stencil']
    device: torch.device
    dtype: torch.dtype
    e: torch.Tensor
    w: torch.Tensor
    cs: torch.Tensor
    equilibrium: 'Equilibrium'

    use_native: bool

    def __init__(self, stencil, device, dtype=torch.float, use_native=True):
        self.stencil = stencil
        self.device = device
        self.dtype = dtype
        self.e = self.convert_to_tensor(stencil.e)
        self.w = self.convert_to_tensor(stencil.w)
        self.cs = self.convert_to_tensor(stencil.cs)
        self.equilibrium = QuadraticEquilibrium(self)
        self.use_native = use_native

    def __str__(self):
        return f"Lattice (stencil {self.stencil.__name__}; device {self.device}; dtype {self.dtype})"

    @property
    def D(self):
        return self.stencil.e.shape[1]

    @property
    def Q(self):
        return self.stencil.e.shape[0]

    def convert_to_tensor(self, array):

        def is_bool_array(it):
            return (isinstance(it, torch.BoolTensor) or
                    (isinstance(it, np.ndarray) and it.dtype in [bool, np.uint8]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if is_bool_array(array):
                return torch.tensor(array, device=self.device, dtype=torch.bool)
            else:
                return torch.tensor(array, device=self.device, dtype=self.dtype)

    @classmethod
    def convert_to_numpy(cls, tensor):
        return tensor.detach().cpu().numpy()

    def rho(self, f):
        """density"""
        return torch.sum(f, dim=0)[None, ...]

    def j(self, f):
        """momentum"""
        return self.einsum("qd,q->d", [self.e, f])

    def u(self, f, rho=None, acceleration=None):
        """velocity; the `acceleration` is used to compute the correct velocity in the presence of a forcing scheme."""
        if rho is None:
            rho = self.rho(f)
        v = self.j(f) / rho
        # apply correction due to forcing, which effectively averages the pre- and post-collision velocity
        correction = 0.0
        if acceleration is not None:
            if len(acceleration.shape) == 1:
                index = [Ellipsis] + [None]*self.D
                acceleration = acceleration[index]
            correction = acceleration / (2 * rho)
        return v + correction

    def incompressible_energy(self, f):
        """incompressible kinetic energy"""
        return 0.5 * self.einsum("d,d->", [self.u(f), self.u(f)])

    def entropy(self, f):
        """entropy according to the H-theorem"""
        f_log = -torch.log(self.einsum("q,q->q", [f, 1 / self.w]))
        return self.einsum("q,q->", [f, f_log])

    def pseudo_entropy_global(self, f):
        """pseudo_entropy derived by a Taylor expansion around the weights"""
        f_w = self.einsum("q,q->q", [f, 1 / self.w])
        return self.rho(f) - self.einsum("q,q->", [f, f_w])

    def pseudo_entropy_local(self, f):
        """pseudo_entropy derived by a Taylor expansion around the local equilibrium"""
        f_feq = f / self.equilibrium(self.rho(f), self.u(f))
        return self.rho(f) - self.einsum("q,q->", [f, f_feq])

    def shear_tensor(self, f):
        """computes the shear tensor of a given f in the sense Pi_{\alpha \beta} = f_i * e_{i \alpha} * e_{i \beta}"""
        shear = self.einsum("qa,qb->qab", [self.e, self.e])
        shear = self.einsum("q,qab->ab", [f, shear])
        return shear

    def mv(self, m, v):
        """matrix-vector multiplication"""
        return self.einsum("ij,j->i", [m, v])

    def einsum(self, equation, fields, **kwargs):
        """Einstein summation on local fields."""
        input, output = equation.split("->")
        inputs = input.split(",")
        for i, inp in enumerate(inputs):
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
