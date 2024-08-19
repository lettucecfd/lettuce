import pickle

import numpy as np
import torch

from typing import Optional, List, Union, Callable, AnyStr
from abc import ABC, abstractmethod

from . import *
from .cuda_native import NativeEquilibrium

__all__ = ['Equilibrium', 'Flow', 'Boundary']


class Equilibrium(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow', rho=None, u=None) -> torch.Tensor:
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self) -> 'NativeEquilibrium':
        ...


class Boundary(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow'):
        ...

    @abstractmethod
    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self, index: int) -> 'NativeBoundary':
        ...


class Flow(ABC):
    """
    The Flow contains physical configurations and state of the simulation.
    """

    # the context can not change
    # during the lifetime of the flow
    # therefore it is stored with it
    context: 'Context'

    # configuration of the flow
    resolution: List[int]
    units: 'UnitConversion'
    stencil: 'Stencil'
    torch_stencil: 'TorchStencil'
    equilibrium: 'Equilibrium'
    boundaries: List['Boundary']

    # current physical state
    i: int
    f: torch.Tensor
    _f_next: Optional[torch.Tensor]

    def __init__(self, context: 'Context', resolution: List[int],
                 units: 'UnitConversion', stencil: 'Stencil',
                 equilibrium: 'Equilibrium'):

        self.context = context
        self.resolution = resolution
        self.units = units
        self.stencil = stencil
        self.torch_stencil = TorchStencil(stencil, context)
        self.equilibrium = equilibrium

        self.i = 0
        self.f = context.empty_tensor([stencil.q, *resolution])
        self._f_next = None

        self.initialize()

    @property
    @abstractmethod
    def boundaries(self) -> List['Boundary']:
        """boundaries"""
        return []

    @abstractmethod
    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        """initial solution in physical units"""
        ...

    def initialize(self):
        """initializing in equilibrium"""
        initial_p, initial_u = self.initial_pu()
        initial_rho = self.context.convert_to_tensor(
            self.units.convert_pressure_pu_to_density_lu(initial_p))
        initial_u = self.context.convert_to_tensor(
            self.units.convert_velocity_to_lu(initial_u))
        self.f = self.context.convert_to_tensor(
            self.equilibrium(self, rho=initial_rho, u=initial_u))

    @property
    def f_next(self) -> torch.Tensor:
        if self._f_next is None:
            # lazy creation of the f_next buffer
            self._f_next = self.context.empty_tensor(
                [self.stencil.q, *self.resolution])
        return self._f_next

    @f_next.setter
    def f_next(self, f_next_: torch.Tensor):
        self._f_next = f_next_

    def rho(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """density"""
        return torch.sum(self.f if f is None else f, dim=0)[None, ...]

    @property
    def rho_pu(self) -> torch.Tensor:
        return self.units.convert_density_to_pu(self.rho())

    @property
    def p_pu(self) -> torch.Tensor:
        return self.units.convert_density_lu_to_pressure_pu(self.rho())

    @property
    def u_pu(self):
        return self.units.convert_velocity_to_pu(self.u())

    def j(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """momentum"""
        return self.einsum("qd,q->d",
                           [self.torch_stencil.e, self.f if f is None else f])

    def u(self, rho=None, acceleration=None) -> torch.Tensor:
        """velocity; the `acceleration` is used to compute the correct velocity
        in the presence of a forcing scheme."""
        rho = self.rho() if rho is None else rho
        v = self.j() / rho
        # apply correction due to forcing, which effectively averages the pre-
        # and post-collision velocity
        if acceleration is None:
            correction = 0.0
        else:
            if len(acceleration.shape) == 1:
                index = [Ellipsis] + [None] * self.stencil.d
                acceleration = acceleration[index]
            correction = acceleration / (2 * rho)
        return v + correction

    @property
    def velocity(self):
        return self.j() / self.rho()

    def incompressible_energy(self, f: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        """incompressible kinetic energy"""
        return 0.5 * self.einsum("d,d->", [self.u(f), self.u(f)])

    def entropy(self) -> torch.Tensor:
        """entropy according to the H-theorem"""
        f_log = -torch.log(self.einsum("q,q->q",
                                       [self.f, 1 / self.torch_stencil.w]))
        return self.einsum("q,q->", [self.f, f_log])

    def pseudo_entropy_global(self) -> torch.Tensor:
        """pseudo_entropy derived by a Taylor expansion around the weights"""
        f_w = self.einsum("q,q->q", [self.f, 1 / self.torch_stencil.w])
        return self.rho() - self.einsum("q,q->", [self.f, f_w])

    def pseudo_entropy_local(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """pseudo_entropy derived by a Taylor expansion around the local
        equilibrium"""
        f = self.f if f is None else f
        f_feq = f / self.equilibrium(self)
        return self.rho(f) - self.einsum("q,q->", [f, f_feq])

    def shear_tensor(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """computes the shear tensor of a given self.f in the sense
        Pi_{\alpha \beta} = f_i * e_{i \alpha} * e_{i \beta}"""
        shear = self.einsum("qa,qb->qab",
                            [self.torch_stencil.e, self.torch_stencil.e])
        shear = self.einsum("q,qab->ab", [self.f if f is None else f, shear])
        return shear

    def mv(self, m, v) -> torch.Tensor:
        """matrix-vector multiplication"""
        return self.einsum("ij,j->i", [m, v])

    def einsum(self, equation, fields, *args) -> torch.Tensor:
        """Einstein summation on local fields."""
        inputs, output = equation.split("->")
        inputs = inputs.split(",")
        for i, inp in enumerate(inputs):
            if len(inp) == len(fields[i].shape):
                pass
            elif len(inp) == len(fields[i].shape) - self.stencil.d:
                inputs[i] += "..."
                if not output.endswith("..."):
                    output += "..."
            else:
                assert False, "Bad dimension."
        equation = ",".join(inputs) + "->" + output
        return torch.einsum(equation, fields, *args)

    def dump(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.context.convert_to_ndarray(self.f), file)

    def load(self, filename):
        with open(filename, "rb") as file:
            self.f = self.context.convert_to_tensor(pickle.load(file),
                                                    dtype=self.context.dtype)

        if self.context.use_native:
            self._f_next = self.context.empty_tensor(self.f.shape)
