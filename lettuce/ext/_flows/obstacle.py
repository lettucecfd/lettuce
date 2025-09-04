import warnings
from typing import Union, List, Optional

import numpy as np
import torch

from . import ExtFlow
from ... import UnitConversion, Context, Stencil, Equilibrium
from ...util import append_axes
from .. import (EquilibriumBoundaryPU, BounceBackBoundary,
                EquilibriumOutletP, AntiBounceBackOutlet)

__all__ = ['Obstacle']


class Obstacle(ExtFlow):
    """
    Flow class to simulate the flow around an object (mask).
    It consists of one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive
    x direction.

    Parameters
    ----------
    resolution : Tuple[int]:
        Grid resolution.
    domain_length_x : float
        Length of the domain in physical units.

    Attributes
    ----------
    _mask : torch.Tensor with dtype = bool
        Boolean mask to define the obstacle. The shape of this object is the
        shape of the grid.
        Initially set to zero (no obstacle).

    Examples
    --------
    Initialization of flow around a cylinder:

    >>> from lettuce import D2Q9
    >>> flow = Obstacle(
    >>>     shape=(101, 51),
    >>>     reynolds_number=100,
    >>>     mach_number=0.1,
    >>>     stencil=D2Q9,
    >>>     domain_length_x=10.1
    >>> )
    >>> x, y = flow.grid
    >>> condition = np.sqrt((x-2.5)**2+(y-2.5)**2) < 1.
    >>> flow.mask[np.where(condition)] = 1
   """

    def __init__(self, context: Context, resolution: Union[int, List[int]],
                 reynolds_number, mach_number, domain_length_x,
                 char_length=1, char_velocity=1,
                 stencil: Optional[Stencil] = None,
                 equilibrium: Optional[Equilibrium] = None):
        self.char_length_lu = resolution[0] / domain_length_x * char_length
        self.char_length = char_length
        self.char_velocity = char_velocity
        self.resolution = self.make_resolution(resolution, stencil)
        self._mask = torch.zeros(self.resolution, dtype=torch.bool)
        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)

    def make_units(self, reynolds_number, mach_number, resolution: List[int]
                   ) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=self.char_length_lu,
            characteristic_length_pu=self.char_length,
            characteristic_velocity_pu=self.char_velocity
        )

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * (stencil.d or self.stencil.d)
        else:
            return resolution

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert ((isinstance(m, np.ndarray) or isinstance(m, torch.Tensor)) and
                all(m.shape[dim] == self.resolution[dim] for dim in range(
                    self.stencil.d)))
        self._mask = self.context.convert_to_tensor(m, dtype=torch.bool)

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        p = np.zeros_like(self.grid[0], dtype=float)[None, ...]
        u_char = self.units.characteristic_velocity_pu * self._unit_vector()
        u_char = append_axes(u_char, self.stencil.d)
        u = ~self.mask * u_char
        return p, u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(torch.arange(n)) for n in
                    self.resolution)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def post_boundaries(self):
        x = self.grid[0]
        return [
            EquilibriumBoundaryPU(flow=self, context=self.context,
                                  mask=torch.abs(x) < 1e-6,
                                  velocity=self.units.
                                  characteristic_velocity_pu
                                  * self._unit_vector()
                                  ),
            AntiBounceBackOutlet(self._unit_vector().tolist(),
                                 self),
            # EquilibriumOutletP(direction=self._unit_vector().tolist(),
            # self, rho_outlet=0),
            BounceBackBoundary(self.mask)
        ]

    def _unit_vector(self, i=0):
        return torch.eye(self.stencil.d)[i]


def Obstacle2D(context: 'Context', resolution: Union[int, List[int]],
               reynolds_number, mach_number, stencil: 'Stencil',
               char_length_lu):
    warnings.warn("Obstacle2D is deprecated. Use Obstacle instead",
                  DeprecationWarning)
    resolution_x = resolution[0] if isinstance(resolution, list) \
        else resolution
    domain_length_x = resolution_x / char_length_lu
    return Obstacle(context=context, resolution=resolution,
                    reynolds_number=reynolds_number, mach_number=mach_number,
                    domain_length_x=domain_length_x, stencil=stencil)


def Obstacle3D(context: 'Context', resolution: Union[int, List[int]],
               reynolds_number, mach_number, stencil: 'Stencil',
               char_length_lu):
    warnings.warn("Obstacle3D is deprecated. Use Obstacle instead",
                  DeprecationWarning)
    resolution_x = resolution[0] if isinstance(resolution, list) \
        else resolution
    domain_length_x = resolution_x / char_length_lu
    return Obstacle(context=context, resolution=resolution,
                    reynolds_number=reynolds_number, mach_number=mach_number,
                    domain_length_x=domain_length_x, stencil=stencil)
