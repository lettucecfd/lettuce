"""
Doubly shear layer in 2D.
Special Inputs & standard value: shear_layer_width = 80, initial_perturbation_magnitude = 0.05
"""

import numpy as np
import numpy.ma as ma
from lettuce.unit import UnitConversion


class DoublyPeriodicShear2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice, shear_layer_width=80, initial_perturbation_magnitude=0.05):
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
        initial_perturbation_magnitude = self.initial_perturbation_magnitude
        shear_layer_width = self.shear_layer_width

        ux = np.tanh(shear_layer_width * (x[1] - 0.25))
        ux[ma.masked_greater(x[1], 0.5).mask] = np.tanh(shear_layer_width * (0.75 - x[1][ma.masked_greater(x[1], 0.5).mask]))
        uy = np.array(initial_perturbation_magnitude * np.sin(2*np.pi*(x[0] + 0.25)))

        # switching to ij/matrix-indexing -> 1st entry: i = -y (i/line index is going down instead of up like y), 2nd entry: x  = j (column index)
        u = np.array([-uy, ux])
        p = np.array([np.zeros(x[0].shape)])
        return p, u

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y)


    @property
    def boundaries(self):
        return []
