"""
Taylor-Green vortex in 2D and 3D
"""

import numpy as np

from lettuce.unit import UnitConversion


class TaylorGreenVortex2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2*np.pi,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        nu = self.units.viscosity_pu
        u = np.array([np.sin(x[0]) * np.cos(x[1]) * np.exp(-2*nu*t), -np.cos(x[0]) * np.sin(x[1]) * np.exp(-2*nu*t)])
        p = 0.25 * (np.cos(2*x[0]) + np.cos(2*x[1]))* np.exp(-4 * nu * t)
        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y)

    @property
    def boundaries(self):
        return {}



