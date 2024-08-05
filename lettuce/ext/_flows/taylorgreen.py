"""
Taylor-Green vortex in 2D and 3D.
"""
import warnings
from typing import Union, List, Optional

import torch

from ... import UnitConversion
from . import ExtFlow

__all__ = ['TaylorGreenVortex', 'TaylorGreenVortex2D', 'TaylorGreenVortex3D']


class TaylorGreenVortex(ExtFlow):
    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None):
        if stencil is None and not isinstance(resolution, list):
            warnings.warn("Requiring information about dimensionality!"
                          " Either via stencil or resolution. Setting "
                          "dimension to 2.", UserWarning)
            self.d = 2
        else:
            self.stencil = stencil() if callable(stencil) else stencil
            self.d = self.stencil.d if self.stencil is not None else len(
                resolution)
        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * self.d
        else:
            assert len(resolution) in [2, 3], ('the resolution of a '
                                               'taylor-green-vortex '
                                               'must be 2 or 3 dimensional!')
            return resolution

    def make_units(self, reynolds_number, mach_number,
                   resolution) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=resolution[0],
            characteristic_length_pu=2 * torch.pi,
            characteristic_velocity_pu=1)

    @property
    def grid(self):
        endpoints = [2 * torch.pi * (1 - 1 / self.resolution[n]) for n in
                     range(self.d)]  # like endpoint=False in np.linspace
        xyz = tuple(torch.linspace(0, endpoints[n],
                                   steps=self.resolution[0],
                                   device=self.context.device,
                                   dtype=self.context.dtype)
                    for n in range(self.d))
        return torch.meshgrid(*xyz, indexing='ij')

    def initial_pu(self) -> (float, float):
        return self.analytic_solution(t=0)

    def analytic_solution(self, t=0):
        grid = self.grid
        nu = self.context.convert_to_tensor(self.units.viscosity_pu)
        if len(self.resolution) == 2:
            u = torch.stack(
                [torch.cos(grid[0])
                 * torch.sin(grid[1])
                 * torch.exp(-2 * nu * t),
                 -torch.sin(grid[0])
                 * torch.cos(grid[1])
                 * torch.exp(-2 * nu * t)])
            p = -torch.stack(
                [0.25 * (torch.cos(2 * grid[0]) + torch.cos(2 * grid[1]))
                 * torch.exp(-4 * nu * t)])
        else:
            u = torch.stack(
                [torch.sin(grid[0])
                 * torch.cos(grid[1])
                 * torch.cos(grid[2]),
                 -torch.cos(grid[0])
                 * torch.sin(grid[1])
                 * torch.cos(grid[2]),
                 torch.zeros_like(grid[0])])
            p = torch.stack(
                [1 / 16. * (torch.cos(2 * grid[0]) + torch.cos(2 * grid[1]))
                 * (torch.cos(2 * grid[2]) + 2)])
        return p, u

    @property
    def boundaries(self) -> List['Boundary']:
        return []


def TaylorGreenVortex3D(context: 'Context', resolution: Union[int, List[int]],
                        reynolds_number, mach_number,
                        stencil: Optional['Stencil'] = None,
                        equilibrium: Optional['Equilibrium'] = None):
    warnings.warn("TaylorGreenVortex3D is deprecated. Use TaylorGreenVortex"
                  " instead", DeprecationWarning)
    return TaylorGreenVortex(context=context, resolution=resolution,
                             reynolds_number=reynolds_number,
                             mach_number=mach_number, stencil=stencil,
                             equilibrium=equilibrium)


def TaylorGreenVortex2D(context: 'Context', resolution: Union[int, List[int]],
                        reynolds_number, mach_number,
                        stencil: Optional['Stencil'] = None,
                        equilibrium: Optional['Equilibrium'] = None):
    warnings.warn("TaylorGreenVortex2D is deprecated. Use TaylorGreenVortex"
                  " instead", DeprecationWarning)
    return TaylorGreenVortex(context=context, resolution=resolution,
                             reynolds_number=reynolds_number,
                             mach_number=mach_number, stencil=stencil,
                             equilibrium=equilibrium)
