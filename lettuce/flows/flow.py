from typing import Tuple

import numpy as np

from lettuce.unit import UnitConversion

from lettuce.lattices import Lattice
from lettuce.boundary import LettuceBoundary


class Grid(np.ndarray):
    def __init__(self):
        super().__init__()
        return


class LettuceFlow:
    D: int
    lattice: Lattice
    shape: Tuple[int, int] or Tuple[int, int, int]
    units: UnitConversion
    grid: Grid
    boundaries: list[LettuceBoundary, ...]

    def __init__(self):
        return

    def initial_solution(self, x: Grid):
        p = np.zeros_like(x[0])
        u_i = np.zeros_like(x[0])
        return p, [u_i, u_i, u_i] if self.D == 3 else [u_i, u_i]

    @property
    def boundaries(self):
        return []

    @staticmethod
    def D(self):
        self._D = len(self.shape) if hasattr(self, 'shape') else self.lattice.D
        return self._D
