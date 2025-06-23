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

    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None):
        self.u0 = 0  # background velocity
        super().__init__(context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * 2
        else:
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution: List[int]
                   ) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=resolution[0],
            characteristic_length_pu=1,
            characteristic_velocity_pu=self.u0
        )

    def analytic_solution(self):
        dvdy = 1/self.resolution[0]
        x, y = self.grid
        u = self.context.convert_to_tensor([dvdy*y + self.u0])
        return u

    def initial_pu(self):
        zeros = self.context.zero_tensor(self.resolution)
        p = zeros[None, ...]
        u = torch.stack([zeros, zeros], dim=0)
        return p, u

    @property
    def grid(self):
        xyz = tuple(torch.linspace(0, 1, steps=n,
                                   device=self.context.device,
                                   dtype=self.context.dtype)
                    for n in self.resolution)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        ktop = torch.zeros(self.resolution, dtype=torch.bool)
        ktop[:, 1] = True
        kbottom = torch.zeros(self.resolution, dtype=torch.bool)
        kbottom[:, -1] = True
        return [
            # moving bounce back top
            EquilibriumBoundaryPU(flow=self, context=self.context, mask=ktop,
                                  velocity=np.array([1.0, 0.0])),
            # bounce back bottom
            BounceBackBoundary(kbottom)
        ]
