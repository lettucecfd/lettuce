import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet2D, AntiBounceBackOutlet3D#\
    #,ZeroGradientOutletRight, ZeroGradientOutletTop, ZeroGradientOutletBottom


class Obstacle2D(object):

    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = None

    def initialise_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array(
            [0 * x[0] + self.units.convert_velocity_to_lu(self.units.characteristic_velocity_pu), 0 * x[1]],
            dtype=float)

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution_y, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    #@property
    #def grid(self):
    #    x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
    #    y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
    #    return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, np.array(
            [self.units.characteristic_velocity_pu, self.units.characteristic_velocity_pu * 0.0])),
                AntiBounceBackOutlet2D(self.grid, self.units.lattice, [1, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)]


class Obstacle3D(object):

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = None

    def initialise_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array(
            [0 * x[0] + self.units.convert_velocity_to_lu(1.0), x[1] * 0, x[2] * 0.05],
            dtype=float)

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution_y, endpoint=False)
        z = np.linspace(0, 1, num=self.resolution_z, endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    #@property
    #def grid(self):
    #    x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
    #    y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
    #    z = np.linspace(0, self.resolution_z / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
    #    return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, np.array(
            [self.units.characteristic_velocity_pu, self.units.characteristic_velocity_pu * 0.0, self.units.characteristic_velocity_pu * 0.0])),
                AntiBounceBackOutlet3D(self.grid, self.units.lattice, [1, 0, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)]

# how those were used by dominik in the obstacle initialisation
# ZeroGradientOutletRight(np.abs(y-1) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
# ZeroGradientOutletBottom(np.abs(x-1) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
# ZeroGradientOutletTop(np.abs(x) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),

class ZeroGradientOutletTop:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[2, :, 0] = f[2, :, 1]
        f[5, :, 0] = f[5, :, 1]
        f[6, :, 0] = f[6, :, 1]
        return f

class ZeroGradientOutletBottom:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[4, :, -1] = f[4, :, -2]
        f[7, :, -1] = f[7, :, -2]
        f[8, :, -1] = f[8, :, -2]
        return f