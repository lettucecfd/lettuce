"""
Poiseuille Flow
"""
from typing import Union, List, Optional

import numpy as np
import torch

from . import ExtFlow
from lettuce._unit import UnitConversion
from lettuce.ext._boundary import BounceBackBoundary

__all__ = ['PoiseuilleFlow2D']


class PoiseuilleFlow2D(ExtFlow):

    def __init__(self, context: 'Context', resolution, reynolds_number,
                 mach_number, initialize_with_zeros=True):
        super().__init__(context, resolution, reynolds_number, mach_number)
        self.initialize_with_zeros = initialize_with_zeros

    def analytic_solution(self, grid):
        half_lattice_spacing = 0.5 / self.resolution[0]
        x, y = grid
        nu = self.units.viscosity_pu
        rho = 1
        u = self.context.convert_to_tensor([
            self.acceleration[0] / (2 * rho * nu)
            * ((y - half_lattice_spacing) * (1 - half_lattice_spacing - y)),
            np.zeros(x.shape)
        ])
        p = self.context.convert_to_tensor(
            [y * 0 + self.units.convert_density_lu_to_pressure_pu(rho)])
        return p, u

    def initial_pu(self):
        if self.initialize_with_zeros:
            p = np.array([0 * self.grid[0]], dtype=float)
            u = np.array([0 * self.grid[0], 0 * self.grid[1]], dtype=float)
            return p, u
        else:
            return self.analytic_solution(self.grid)

    def make_units(self, reynolds_number, mach_number,
                   resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        pass

    @property
    def grid(self):
        xyz = tuple(torch.linspace(0, 1, steps=n,
                                   device=self.context.device,
                                   dtype=self.context.dtype)
                    for n in self.resolution)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        mask = self.context.zero_tensor(self.grid[0].shape, dtype=bool)
        mask[:, [0, -1]] = True
        boundary = BounceBackBoundary(mask=mask)
        return [boundary]

    @property
    def acceleration(self):
        return self.context.convert_to_tensor([0.001, 0])
