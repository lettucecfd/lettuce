"""
Taylor-Green vortex in 2D and 3D.
"""
from typing import Union, List, Optional

import torch
import numpy as np

from ... import UnitConversion
from . import ExtFlow

__all__ = ['TaylorGreenVortex2D', 'TaylorGreenVortex3D']


class TaylorGreenVortex2D(ExtFlow):
    def __init__(self, context: 'Context', resolution: Union[int, List[int]], reynolds_number, mach_number, stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None):
        ExtFlow.__init__(self, context, resolution, reynolds_number, mach_number, stencil, equilibrium)

    def make_resolution(self, resolution: Union[int, List[int]], stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * 2
        else:
            assert len(resolution) == 2, 'the resolution of a 2d taylor green vortex must obviously be 2!'
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=resolution[0],
            characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=1)

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution[0], endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution[1], endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    def initial_pu(self) -> (float, float):
        return self.analytic_solution(t=0)

    def analytic_solution(self, t=0):
        grid = self.grid
        nu = self.units.viscosity_pu
        u = np.array([np.cos(grid[0]) * np.sin(grid[1]) * np.exp(-2 * nu * t),
                      -np.sin(grid[0]) * np.cos(grid[1]) * np.exp(-2 * nu * t)])
        p = -np.array([0.25 * (np.cos(2 * grid[0]) + np.cos(2 * grid[1])) * np.exp(-4 * nu * t)])
        return p, u


class TaylorGreenVortex3D(ExtFlow):
    def make_resolution(self, resolution: Union[int, List[int]], stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * 3
        else:
            assert len(resolution) == 3, 'the resolution of a 3d taylor green vortex must obviously be 3!'
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution[0] / (2 * np.pi),
            characteristic_length_pu=1,
            characteristic_velocity_pu=1)

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution[0], endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution[1], endpoint=False)
        z = np.linspace(0, 2 * np.pi, num=self.resolution[2], endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    def initial_pu(self):
        return self.analytic_solution()

    def analytic_solution(self):
        grid = self.grid
        u = np.array([
            np.sin(grid[0]) * np.cos(grid[1]) * np.cos(grid[2]),
            -np.cos(grid[0]) * np.sin(grid[1]) * np.cos(grid[2]),
            np.zeros_like(np.sin(grid[0]))])
        p = np.array([1 / 16. * (np.cos(2 * grid[0]) + np.cos(2 * grid[1])) * (np.cos(2 * grid[2]) + 2)])
        return p, u
