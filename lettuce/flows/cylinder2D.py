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
        # komisches Zeug von Mario: u[2] += np.sin(x[2] / x[2].shape[1] * 2 * np.pi) * np.sin(x[0]/x[0].shape[0]*np.pi) * self.units.characteristic_velocity_pu * 0.05

        # Sinus-Störung in ux
        ny = x[1].shape[1]
        u[0][1] += np.sin(np.arange(0, ny) / ny * 2 * np.pi) * self.units.characteristic_velocity_pu * 0.3

        # Block-Störung unten rechts
        #u[0][3:25, 3:50] *= 1.3
        #u[1][3:25, 3:50] *= 1.3
        return p, u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(np.arange(n)) for n in self.shape)
        return np.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        x = self.grid[0]
        outmask = np.zeros(self.grid[0].shape, dtype=bool)
        outmask[[0, -1], :] = True
        return [
            EquilibriumBoundaryPU(
                # np.abs(x) < 1e-6,
                outmask,
                self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            # AntiBounceBackOutlet(self.units.lattice, self._unit_vector().tolist()),
            # EquilibriumOutletP(), # wird von PyCharm als "unknown reference" markiert?
            # EquilibriumBoundaryPU(
            BounceBackBoundary(self.mask, self.units.lattice)
        ]

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]