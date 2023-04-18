
import warnings
import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet


class Obstacle:
    """
    Flow class to simulate the flow around an object (mask).
    It consists of one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

    Parameters
    ----------
    shape : Tuple[int]:
        Grid resolution.
    domain_length_x : float
        Length of the domain in physical units.

    Attributes
    ----------
    mask : np.array with dtype = bool
        Boolean mask to define the obstacle. The shape of this object is the shape of the grid.
        Initially set to zero (no obstacle).

    Examples
    --------
    Initialization of flow around a cylinder:

    >>> from lettuce import Lattice, D2Q9
    >>> flow = Obstacle(
    >>>     shape=(101, 51),
    >>>     reynolds_number=100,
    >>>     mach_number=0.1,
    >>>     lattice=lattice,
    >>>     domain_length_x=10.1
    >>> )
    >>> x, y = flow.grid
    >>> condition = np.sqrt((x-2.5)**2+(y-2.5)**2) < 1.
    >>> flow.mask[np.where(condition)] = 1
   """
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
        self._mask = np.zeros(shape=self.shape, dtype=bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.shape
        self._mask = m.astype(bool)

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
        x = self.grid[0]
        return [
            EquilibriumBoundaryPU(
                np.abs(x) < 1e-6, self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            AntiBounceBackOutlet(self.units.lattice, self._unit_vector().tolist()),
            BounceBackBoundary(self.mask, self.units.lattice)
        ]

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]


def Obstacle2D(resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu):
    warnings.warn("Obstacle2D is deprecated. Use Obstacle instead", DeprecationWarning)
    shape = (resolution_x, resolution_y)
    domain_length_x = resolution_x / char_length_lu
    return Obstacle(shape, reynolds_number, mach_number, lattice, domain_length_x=domain_length_x)


def Obstacle3D(resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu):
    warnings.warn("Obstacle3D is deprecated. Use Obstacle instead", DeprecationWarning)
    shape = (resolution_x, resolution_y, resolution_z)
    domain_length_x = resolution_x / char_length_lu
    return Obstacle(shape, reynolds_number, mach_number, lattice, domain_length_x=domain_length_x)
