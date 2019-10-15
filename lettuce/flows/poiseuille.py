"""
Couette Flow
"""

import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU
import matplotlib.pyplot as plt


class PoiseuilleFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.lattice = lattice
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, grid, dpdx=0, Fx=0):
        x = grid[1] * (grid[0].shape[1] - 1)
        nu = self.units.viscosity_lu
        rho = 1
        u = np.array([-Fx / (2 * rho * nu) * ((x - x.max() / 2) ** 2 - (x.max() / 2) ** 2), np.zeros(grid[1].shape)])
        print("t0: ux: %.10f " % u[0].max() + "uy: %.10f" % u[1].max())
        p = np.array([x * 0 + self.units.convert_density_lu_to_pressure_pu(rho)])

        plt.plot(grid[1][2] * (grid[0].shape[1] - 1), u[0][2])

        return p, u

    def initial_solution(self, grid):
        return self.analytic_solution(grid, Fx=1e-5)

    @property
    def grid(self):
        x = np.linspace(0, 1, num=51, endpoint=True)
        y = np.linspace(0, 1, num=51, endpoint=True)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        mask = np.zeros(self.grid[0].shape, dtype=bool)
        mask[:, [0, -1]] = True
        boundary = BounceBackBoundary(mask=mask, lattice=self.units.lattice)
        return [boundary]

    @property
    def F(self):
        F = torch.tensor([1e-5, 0.0])
        return F

