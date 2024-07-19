"""
Couette Flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU
from lettuce.flows.flow import LettuceFlow


class CouetteFlow2D(LettuceFlow):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        super().__init__()
        self.resolution = resolution
        self.shape = (resolution, resolution)
        self._mask = np.zeros(shape=self.shape, dtype=bool)
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.u0 = 0  # background velocity

    def analytic_solution(self, x):
        dvdy = 1/self.resolution
        x, y = self.grid
        u = np.array([dvdy*y + self.u0], dtype=float)
        return u

    def initial_solution(self, x):
        return (np.array([0 * x[0]], dtype=float),
                np.array([0 * x[0], 0 * x[1]], dtype=float))

    @property
    def mask(self):
        return self._mask

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
            EquilibriumBoundaryPU(ktop, self.units.lattice, self.units,
                                  np.array([1.0, 0.0])),
            # bounce back bottom
            BounceBackBoundary(kbottom, self.units.lattice)
        ]
