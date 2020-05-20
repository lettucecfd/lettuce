"""
Doubly shear layer in 2D.
"""

import numpy as np
from lettuce.unit import UnitConversion


class DoublyPeriodicShear2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x, shear_layer_width = 80, initial_perturbation_magnitude = 0.05):
        # x direction ?
        u1 = np.concatenate((np.tanh(shear_layer_width * (x[1][0:int(np.floor(x[1].shape[0]/2)), :] - 0.25)),
                             np.tanh(shear_layer_width * (0.75 - x[1][int(np.floor(x[1].shape[0]/2)):, :]))))
        # y direction ?
        u2 = np.array(initial_perturbation_magnitude * np.sin(2*np.pi*(x[0] + 0.25)))
        u = np.array([-u2, u1])
        p = np.array([np.zeros(x[0].shape)])
        return p, u

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y)#, indexing = 'ij')

    @property
    def boundaries(self):
        return []
