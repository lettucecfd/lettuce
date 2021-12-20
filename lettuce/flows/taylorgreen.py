"""
Taylor-Green vortex in 2D and 3D.
"""

import numpy as np

from lettuce.unit import UnitConversion


class TaylorGreenVortex2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        nu = self.units.viscosity_pu
        u = np.array([np.cos(x[0]) * np.sin(x[1]) * np.exp(-2 * nu * t),
                      -np.sin(x[0]) * np.cos(x[1]) * np.exp(-2 * nu * t)])
        p = -np.array([0.25 * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * np.exp(-4 * nu * t)])
        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []


class TaylorGreenVortex3D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution / (2 * np.pi), characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):
        u = np.array([
            np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2]),
            -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2]),
            np.zeros_like(np.sin(x[0]))
        ])
        p = np.array([1 / 16. * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * (np.cos(2 * x[2]) + 2)])
        return p, u

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        z = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        return []
