"""
Couette Flow
"""
from typing import Union, List, Optional

import numpy as np
import torch

from ... import UnitConversion
from .. import BounceBackBoundary, EquilibriumBoundaryPU
from ._ext_flow import ExtFlow

__all__ = ['CouetteFlow2D']


class CouetteFlow2D(ExtFlow):

    def __init__(self, resolution, reynolds_number, mach_number, lattice,
                 context: 'Context'):
        super().__init__(context, resolution, reynolds_number, mach_number)
        self.resolution = resolution
        self.units = UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.u0 = 0  # background velocity

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * (stencil.d or self.stencil.d)
        else:
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution: List[int]
                   ) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=resolution[0],
            characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self):
        dvdy = 1/self.resolution[0]
        x, y = self.grid
        u = self.context.convert_to_tensor([dvdy*y + self.u0])
        return u

    def initial_solution(self, x):
        return (torch.zeros(self.resolution[0]),
                torch.zeros(self.resolution))

    @property
    def grid(self):
        x = torch.linspace(0, 1, self.resolution[0])
        y = torch.linspace(0, 1, self.resolution[1])
        return torch.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        ktop = np.zeros(np.shape(y), dtype=bool)
        ktop[:, -1] = True
        kbottom = np.zeros(np.shape(y), dtype=bool)
        kbottom[:, 1] = True
        return [
            # moving bounce back top
            EquilibriumBoundaryPU(ktop, self.units.lattice, self.units,
                                  np.array([1.0, 0.0])),
            # bounce back bottom
            BounceBackBoundary(kbottom, self.units.lattice)
        ]
