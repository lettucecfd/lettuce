import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary#, ZeroGradientOutletRight, ZeroGradientOutletTop, \
    #ZeroGradientOutletBottom


class Obstacle2D(object):

    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution_x * 0.1, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = 0

    def initizialize_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array(
            [0 * x[0] + self.units.characteristic_velocity_lu, 0.05 * self.units.characteristic_velocity_lu * x[1]],
            dtype=float)

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution_y, endpoint=False)
        return np.meshgrid(x, y)

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(x - 1) < 1e-6, self.units.lattice, self.units, np.array(
            [self.units.characteristic_velocity_pu, self.units.characteristic_velocity_pu * 0.01])),
                ZeroGradientOutletRight(np.abs(x) < 1e-6, self.units.lattice, direction=[1.0, 0.0]),
                BounceBackBoundary(self.mask, self.units.lattice),
                ZeroGradientOutletTop(np.abs(x) < 1e-6, self.units.lattice, direction=[1.0, 0.0]),
                ZeroGradientOutletBottom(np.abs(x) < 1e-6, self.units.lattice, direction=[1.0, 0.0])]


class ZeroGradientOutletRight:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[:, -1] = f[:, -2]
        return f


class ZeroGradientOutletTop:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[:, :, 0] = f[:, :, 1]
        return f


class ZeroGradientOutletBottom:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[:, :, -1] = f[:, :, -2]
        return f