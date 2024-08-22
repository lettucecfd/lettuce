import pickle

import numpy as np
import torch

from typing import Optional, List, Union, Callable, AnyStr
from abc import ABC, abstractmethod

from . import TorchStencil
from .util import torch_gradient, torch_jacobi
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
    initialize_pressure: bool = False
    initialize_fneq: bool = False

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
        if self.initialize_pressure:
            initial_rho = pressure_poisson(
                self.units,
                initial_u,
                initial_rho
            )
            self.f = self.equilibrium(self, rho=initial_rho, u=initial_u)
        self.f = self.equilibrium(self, rho=initial_rho, u=initial_u)
        if self.initialize_fneq:
            self.f = initialize_f_neq(self)

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

    def u(self, f: Optional[torch.Tensor] = None, rho=None, acceleration=None
          ) -> torch.Tensor:
        """velocity; the `acceleration` is used to compute the correct velocity
        in the presence of a forcing scheme."""
        rho = self.rho(f=f) if rho is None else rho
        v = self.j(f=f) / rho
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

    def pseudo_entropy_local(self, f: Optional[torch.Tensor] = None
                             ) -> torch.Tensor:
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


def pressure_poisson(units: 'UnitConversion', u, rho0, tol_abs=1e-10,
                     max_num_steps=100000):
    """
    Solve the pressure poisson equation using a jacobi scheme.

    Parameters
    ----------
    units : lettuce.UnitConversion
        The flow instance.
    u : torch.Tensor
        The velocity tensor.
    rho0 : torch.Tensor
        Initial guess for the density (i.e., pressure).
    tol_abs : float
        The tolerance for pressure convergence.


    Returns
    -------
    rho : torch.Tensor
        The converged density (i.e., pressure).
    """
    # convert to physical units
    dx = units.convert_length_to_pu(1.0)
    u = units.convert_velocity_to_pu(u)
    p = units.convert_density_lu_to_pressure_pu(rho0)

    # compute laplacian
    with torch.no_grad():
        u_mod = torch.zeros_like(u[0])
        dim = u.shape[0]
        for i in range(dim):
            for j in range(dim):
                derivative = torch_gradient(
                    torch_gradient(u[i] * u[j], dx)[i],
                    dx
                )[j]
                u_mod -= derivative
    # TODO(@MCBs): still not working in 3D

    p_mod = torch_jacobi(
        u_mod,
        p[0],
        dx,
        dim=2,
        tol_abs=tol_abs,
        max_num_steps=max_num_steps
    )[None, ...]

    return units.convert_pressure_pu_to_density_lu(p_mod)


def initialize_pressure_poisson(flow: 'Flow',
                               max_num_steps=100000,
                               tol_pressure=1e-6):
    """Reinitialize equilibrium distributions with pressure obtained by a
    Jacobi solver. Note that this method has to be called before
    initialize_f_neq.
    """
    u = flow.u()
    rho = pressure_poisson(
        flow.units,
        u,
        flow.rho(),
        tol_abs=tol_pressure,
        max_num_steps=max_num_steps
    )
    return flow.equilibrium(flow=flow, rho=rho, u=u)


def initialize_f_neq(flow: 'Flow'):
    """Initialize the distribution function values. The f^(1) contributions are
    approximated by finite differences. See Krüger et al. (2017).
    """
    rho = flow.rho()
    u = flow.u()

    grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
    grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
    S = torch.cat([grad_u0, grad_u1])

    if flow.stencil.d == 3:
        grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
        S = torch.cat([S, grad_u2])

    Pi_1 = (1.0 * flow.units.relaxation_parameter_lu * rho * S
            / flow.torch_stencil.cs ** 2)
    print(flow.torch_stencil.e.device)
    Q = (torch.einsum('ia,ib->iab',
                      [flow.torch_stencil.e, flow.torch_stencil.e])
         - torch.eye(flow.stencil.d, device=flow.torch_stencil.e.device)
         * flow.stencil.cs ** 2)
    Pi_1_Q = flow.einsum('ab,iab->i', [Pi_1, Q])
    fneq = flow.einsum('i,i->i', [flow.torch_stencil.w, Pi_1_Q])

    feq = flow.equilibrium(flow, rho, u)

    return feq - fneq
