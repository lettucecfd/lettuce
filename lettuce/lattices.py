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

    field_index = 0

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
        try:
            array = np.moveaxis(array, 0, self.field_index)
        except: array = array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(array, np.ndarray) and array.dtype in [np.bool, np.uint8]:
                return torch.tensor(array, device=self.device, dtype=torch.uint8)  # that's how torch stores its masks
            else:
                return torch.tensor(array, device=self.device, dtype=self.dtype)

    @classmethod
    def convert_to_numpy(cls, tensor):
        return np.moveaxis(tensor.detach().cpu().numpy(), cls.field_index, 0)

    def rho(self, f):
        """density"""
        return torch.sum(f, dim=self.field_index)[self.field(None)]

    def j(self, f):
        """momentum"""
        return self.einsum("qd,q->d", [self.e, f])

    def u(self, f):
        """velocity"""
        return self.j(f) / self.rho(f)

    def field(self, index=None):
        """Generate indices for multidimensional fields.

        All lattice fields are stored as tensors of dimension [M, Nx, Ny, Nz], (in 3D),
        where N... are the grid dimensions and M depends on the quantity
        (density: M=1, velocity: M=D, distribution functions: M=Q).

        Note that one-dimensional quantities such as density are NOT stored in the shape [Nx, Ny, Nz],
        but [1, Nx, Ny, Nz]. lattice.field is used to transform between these two shapes.

        Parameters
        ----------
        index: int or None
            If None, transform a tensor of elements to a field: [Nx, Ny, Nz] -> [1, Nx, Ny, Nz].
            If int, get the i-th element from a field: [M, Nx, Ny, Nz] -> [Nx, Ny, Nz].

        Returns
        -------
        indices: (Multiindex)
            An index for a multidimensional array.

        Notes
        -----
        This method is important to allow different underlying storage orders and support LatticeAoS.

        Examples
        --------
        >>> lattice = Lattice(D2Q9, "cpu")
        >>> f = torch.ones(9,16,16)
        >>> f0 = f[lattice.field(0)]  # -> shape [Nx, Ny]
        >>> rho = torch.sum(f, dim=lattice.field_index) # -> shape [Nx, Ny]
        >>> rho = rho[lattice.field()] # -> shape [1, Nx, Ny]
        """
        return index, Ellipsis

    #def moment(self, f, multiindex):
    #    return torch.einsum("q,q...->...",moment_tensor(self.e, multiindex), f)

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


class LatticeAoS(Lattice):
    """
    Lattice class with inverse storage order (array of structure).
    """

    field_index = -1

    def __init__(self, stencil, device, dtype=torch.float):
        super(LatticeAoS,self).__init__(stencil, device, dtype)

    def __str__(self):
        return f"LatticeOfArray (stencil {self.stencil.__name__}; device {self.device}; dtype {self.dtype})"

    def field(self, index=None):
        return Ellipsis, index

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
            inputs[i] = inputs[i][::-1]
        equation = ",".join(inputs) + "->" + output[::-1]
        return torch.einsum(equation, fields, **kwargs)
