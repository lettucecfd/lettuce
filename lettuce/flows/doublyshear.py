"""
Doubly shear layer in 2D.
Special Inputs & standard value: shear_layer_width = 80, initial_perturbation_magnitude = 0.05
"""

import numpy as np
from lettuce.unit import UnitConversion


class DoublyPeriodicShear2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice, shear_layer_width=80,
                 initial_perturbation_magnitude=0.05):
        self.initial_perturbation_magnitude = initial_perturbation_magnitude
        self.shear_layer_width = shear_layer_width
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        raise NotImplementedError

    def initial_solution(self, x):
        pert = self.initial_perturbation_magnitude
        w = self.shear_layer_width
        u1 = np.choose(
            x[1] > 0.5,
            [np.tanh(w * (x[1] - 0.25)), np.tanh(w * (0.75 - x[1]))]
        )
        u2 = pert * np.sin(2 * np.pi * (x[0] + 0.25))
        u = np.stack([u1, u2], axis=0)
        p = np.zeros_like(u1[None, ...])
        return p, u

    @property
    def grid(self):
        x = np.linspace(0., 1., num=self.resolution, endpoint=False)
        y = np.linspace(0., 1., num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []
