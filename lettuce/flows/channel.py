"""
Cannel flows in 2D.
"""

import numpy as np
import torch
from lettuce.boundary import BounceBackBoundary
from lettuce.unit import UnitConversion


class ChannelFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2*np.pi,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        u = np.array([x[0]*0.0, x[1]*0.0])
        p = np.array([x[0]*0.0+0.33])
        #p[0,:,0:70] = 0.35
        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)


    @property
    def F(self):
        F = torch.tensor([0.00005, 0.0]) #, device=lattice.device, dtype=lattice.dtype
        return F

    @property
    def grid(self):
        x = np.arange(500)
        y = np.arange(60)
        #z = 1
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        mask = np.zeros(self.grid[0].shape, dtype=bool)
        mask[:, [0,-1]] = True
        mask[80:120, 10:50] = True
        return [BounceBackBoundary(mask=mask, lattice=self.units.lattice)]
