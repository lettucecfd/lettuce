"""
Taylor-Green vortex in 2D and 3D.
"""

import numpy as np
from lettuce.unit import UnitConversion
from lettuce.flows.flow import Flow


class TaylorGreenVortex2D(Flow):
    def __init__(self,
                 domain,
                 reynolds_number,
                 mach_number,
                 lattice,
                 compute_f=False):
        self.resolution = domain.shape[1]
        self.units = UnitConversion(
            lattice=lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=self.resolution,
            characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=1
        )
        super().__init__(domain=domain,
                         units=self.units,
                         compute_f=compute_f)

    def analytic_solution(self, x, t=0):
        nu = self.units.viscosity_pu
        u = np.array([np.cos(x[0]) * np.sin(x[1]) * np.exp(-2 * nu * t),
                      -np.sin(x[0]) * np.cos(x[1]) * np.exp(-2 * nu * t)])
        p = -np.array([0.25 * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * np.exp(-4 * nu * t)])
        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)


class TaylorGreenVortex3D(Flow):
    def __init__(self,
                 domain,
                 reynolds_number,
                 mach_number,
                 lattice,
                 compute_f=False):
        self.resolution = domain.shape[1]
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=self.resolution / (2 * np.pi), characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        super().__init__(domain=domain,
                         units=self.units,
                         compute_f=compute_f)

    def initial_solution(self, x):
        u = np.array([
            np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2]),
            -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2]),
            np.zeros_like(np.sin(x[0]))
        ])
        p = np.array([1 / 16. * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * (np.cos(2 * x[2]) + 2)])
        return p, u

    # @property
    # def grid(self):
    #     x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
    #     y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
    #     z = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
    #     return np.meshgrid(x, y, z, indexing='ij')
    #
    # @property
    # def boundaries(self):
    #     return []
