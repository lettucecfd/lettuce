import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary#, ZeroGradientOutletRight, ZeroGradientOutletTop, \
    #ZeroGradientOutletBottom


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

    def initizialize_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array(
            [0 * x[0] + self.units.convert_velocity_to_lu(1.0), x[1]*0],
            dtype=float)

    def getMaxU(self, f, lattice):
        u0 = (lattice.u(f)[0])
        u1 = (lattice.u(f)[1])
        return torch.max(torch.sqrt(u0*u0+u1*u1))

    def getSheddingFrequency(self, sample):
        sample = np.asarray(sample)
        x = np.arange(sample.size)
        fourier = np.fft.fft(sample)
        return x[(fourier == np.min(fourier[2:int(sample.size/2)]))]/ sample.size


    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution_y, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, np.array(
            [self.units.characteristic_velocity_pu, self.units.characteristic_velocity_pu * 0.0])),
                ZeroGradientOutletRight(np.abs(y-1) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
                BounceBackBoundary(self.mask, self.units.lattice)]


class ZeroGradientOutletRight:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[3, -1] = f[3, -2]
        f[6, -1] = f[6, -2]
        f[7, -1] = f[7, -2]
        return f
