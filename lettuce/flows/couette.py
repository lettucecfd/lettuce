"""
Couette Flow
"""

import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU


class CouetteFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        raise NotImplementedError
        nu = self.units.viscosity_pu
        u = None #np.array([np.sin(x[0]) * np.cos(x[1]) * np.exp(-2*nu*t), -np.cos(x[0]) * np.sin(x[1]) * np.exp(-2*nu*t)])
        p = None #0.25 * (np.cos(2*x[0]) + np.cos(2*x[1]))* np.exp(-4 * nu * t)
        return p, u

    def initial_solution(self, x):
        return np.array([0*x[0]], dtype=float), np.array([0*x[0],0*x[1]], dtype=float)

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(y-1) < 1e-6, self.units.lattice, self.units, np.array([1.0, 0.0])),
                BounceBackBoundary(np.abs(y) < 1e-6, self.units.lattice)]



