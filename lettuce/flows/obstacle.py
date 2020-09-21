import numpy as np
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet


class Obstacle2D(object):
    """
    Flow class to simulate the flow around an object (mask) in 2D.
    It consists off one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

    Parameters
    ----------
    resolution_x : int
        Grid resolution in streamwise direction.
    resolution_y : int
        Grid resolution in spanwise direction.
    char_length_u : float
        The characteristic length in lattice units; usually the number of grid points for the obstacle in flow direction

    Attributes
    ----------
    mask : np.array with dtype = np.bool
        Boolean mask to define the obstacle. The shape of this object is the shape of the grid.
        Initially set to zero (no obstacle).

    Examples
    --------
    Initialization of flow around a square:
    >>> from lettuce import Lattice, D2Q9
    >>> lattice = Lattice(D2Q9)
    >>> flow = Obstacle2D(100, 50, reynolds_number=100, mach_number=0.1, lattice=lattice, char_length_lu=5)
    >>> x, y = flow.grid
    >>> np.where(x**2+y**2) <= 1.0
   """
    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y), dtype=np.bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y)
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_char = np.array([self.units.characteristic_velocity_pu, 0.0])[..., None, None]
        u = self.mask.astype(np.float) * u_char
        return p, u

    @property
    def grid(self):
        x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units,
                                      np.array([self.units.characteristic_velocity_pu, 0])),
                AntiBounceBackOutlet(self.units.lattice, [1, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)]


class Obstacle3D(object):
    """Flow class to simulate the flow around an object (mask) in 3D.
    It consists off one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary),
    leading to a flow in positive x direction.

    Attributes
    ----------
    add object mask directly or via "initialize_object" as bool tensor / bool array with true entries forming the object
    char_length_lu: length of the object in flow direction (positive x)"""

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
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=np.bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y, self.resolution_z)
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        u = self.mask.astype(np.float) * u_char
        return p, u

    @property
    def grid(self):
        x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
        z = np.linspace(0, self.resolution_z / self.units.characteristic_length_lu, num=self.resolution_z, endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units,
                                      np.array([self.units.characteristic_velocity_pu, 0, 0])),
                AntiBounceBackOutlet(self.units.lattice, [1, 0, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)]
