"""
Poiseuille Flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary


class PoiseuilleFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice, initialize_with_zeros=True):
        self.resolution = resolution
        self.lattice = lattice
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.initialize_with_zeros = initialize_with_zeros

    def analytic_solution(self, grid):
        half_lattice_spacing = 0.5 / self.resolution
        x, y = grid
        nu = self.units.viscosity_pu
        rho = 1
        u = np.array([
            self.acceleration[0] / (2 * rho * nu) * ((y - half_lattice_spacing) * (1 - half_lattice_spacing - y)),
            np.zeros(x.shape)
        ])
        p = np.array([y * 0 + self.units.convert_density_lu_to_pressure_pu(rho)])
        return p, u

    def initial_solution(self, grid):
        if self.initialize_with_zeros:
            p = np.array([0 * grid[0]], dtype=float)
            u = np.array([0 * grid[0], 0 * grid[1]], dtype=float)
            return p, u
        else:
            return self.analytic_solution(grid)

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution + 1, endpoint=True)
        y = np.linspace(0, 1, num=self.resolution + 1, endpoint=True)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        mask = np.zeros(self.grid[0].shape, dtype=bool)
        mask[:, [0, -1]] = True
        boundary = BounceBackBoundary(mask=mask, lattice=self.units.lattice)
        return [boundary]

    @property
    def acceleration(self):
        return np.array([0.001, 0])
