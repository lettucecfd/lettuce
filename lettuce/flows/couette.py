"""
Couette Flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU


class CouetteFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice, v_top=1.0, v_bottom=0.0):
        self.resolution = resolution
        self.shape = (resolution, resolution)
        self._mask = np.zeros(shape=self.shape, dtype=bool)
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.velocities = np.array([v_top, v_bottom])

    def analytic_solution(self, x):
        v1 = self.velocities[0]
        v0 = self.velocities[1]
        dvdy = (v1-v0)/self.resolution
        x, y = self.grid
        u = np.array([dvdy*y + v0], dtype=float)
        return u

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array([0 * x[0], 0 * x[1]], dtype=float)

    @property
    def mask(self):
        return self._mask\

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        ktop = np.zeros(np.shape(y), dtype=bool)
        ktop[:, -1] = True
        kbottom = np.zeros(np.shape(y), dtype=bool)
        kbottom[:, 1] = True
        return [
            # moving bounce back top
            EquilibriumBoundaryPU(ktop, self.units.lattice, self.units, self.velocities),
            # bounce back bottom
            BounceBackBoundary(kbottom, self.units.lattice)
        ]
