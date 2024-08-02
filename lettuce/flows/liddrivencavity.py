"""
Cavity flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU
from lettuce.flows.flow import LettuceFlow


class Cavity2D(LettuceFlow):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.shape = (resolution, resolution)
        self._mask = np.zeros(shape=self.shape, dtype=bool)
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):
        return (np.array([0 * x[0]], dtype=float),
                np.array([0 * x[0], 0 * x[1]], dtype=float))  # p, u

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
        boundary = np.zeros(np.shape(y), dtype=bool)
        top = np.zeros(np.shape(y), dtype=bool)
        boundary[[0, -1], 1:] = True  # left and right
        boundary[:, 0] = True  # bottom
        top[:, -1] = True  # top
        return [
            # bounce back walls
            BounceBackBoundary(boundary, self.units.lattice),
            # moving fluid on top# moving bounce back top
            EquilibriumBoundaryPU(top, self.units.lattice, self.units,
                                  np.array([1.0, 0.0])),
        ]
