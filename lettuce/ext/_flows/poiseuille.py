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

from .._stencil import D2Q9


class PoiseuilleFlow2D(ExtFlow):

    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 initialize_with_zeros=True):
        self.stencil = D2Q9() if stencil is None else stencil
        self.initialize_with_zeros = initialize_with_zeros
        super().__init__(context, resolution, reynolds_number, mach_number,
                         self.stencil, equilibrium)

    def analytic_solution(self, t=0) -> (torch.Tensor, torch.Tensor):
        half_lattice_spacing = 0.5 / self.resolution[0]
        x, y = self.grid
        nu = self.units.viscosity_pu
        rho = 1
        ux = (self.acceleration[0] / (2 * rho * nu)
              * ((y - half_lattice_spacing) * (1 - half_lattice_spacing - y)))
        uy = self.context.zero_tensor(self.resolution)
        u = torch.stack([ux, uy], dim=0)
        p = y * 0 + self.units.convert_density_lu_to_pressure_pu(rho)
        return p, u

    def initial_pu(self):
        if self.initialize_with_zeros:
            zeros = self.context.zero_tensor(self.resolution)
            p = zeros[None, ...]
            u = torch.stack(2*[zeros], dim=0)
            return p, u
        else:
            return self.analytic_solution()

    def make_units(self, reynolds_number, mach_number,
                   resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution[0]-1,
            characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, list):
            assert len(resolution) == self.stencil.d
        if isinstance(resolution, int):
            resolution = [resolution] * self.stencil.d
        return resolution

    @property
    def grid(self):
        xyz = tuple(torch.linspace(0, 1,
                                   steps=n,
                                   device=self.context.device,
                                   dtype=self.context.dtype)
                    for n in self.resolution)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def post_boundaries(self):
        mask = self.context.zero_tensor(self.resolution, dtype=bool)
        mask[:, [0, -1]] = True
        boundary = BounceBackBoundary(mask=mask)
        return [boundary]

    @property
    def acceleration(self):
        return self.context.convert_to_tensor([0.001, 0])
