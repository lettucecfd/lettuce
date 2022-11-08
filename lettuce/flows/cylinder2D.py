import warnings
import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet


class Cylinder2D:

    def __init__(self, shape, reynolds_number, mach_number, lattice, domain_length_x, char_length=1, char_velocity=1):
        if len(shape) != lattice.D:
            raise ValueError(f"{lattice.D}-dimensional lattice requires {lattice.D}-dimensional `shape`")
        self.shape = shape
        char_length_lu = shape[0] / domain_length_x * char_length
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length,
            characteristic_velocity_pu=char_velocity
        )
        self._mask = np.zeros(shape=self.shape, dtype=np.bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.shape
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_char = self.units.characteristic_velocity_pu * self._unit_vector()
        u_char = append_axes(u_char, self.units.lattice.D)
        u = (1 - self.mask) * u_char
        return p, u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(np.arange(n)) for n in self.shape)
        return np.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        return []
