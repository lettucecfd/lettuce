"""
Cannel flows in 2D.
"""

import numpy as np

from lettuce.unit import UnitConversion


class ChannelFlow2D(object):
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
        u = np.array([x[0]*0.0, x[1]*0.0])
        #u[0,1:50,190:210] = 0.2
        #u[1, 90:110, 190:210] = 0.2
        p = np.array([x[0]*0.0+0.33])
        p[0,:,0:100] = 0.35
        #p[0, 0:240, :] = 0.33
        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)

    @property
    def grid(self):
        x = np.arange(self.resolution*2)
        y = np.arange(self.resolution)
        return np.meshgrid(x, y)

    @property
    def boundaries(self):
        print('test')
        return []
