from abc import abstractmethod
from typing import Union
import numpy as np

from . import LatticeBase


class PipeBase(LatticeBase):
    """Class PipeBase
    """

    step_begin = False
    step_end = False

    def __init__(self, lattice):
        """Trivial Constructor for Class PipeBase

        Parameters
        ----------
        lattice: Lattice
            The associated lattice object.
            Used for configuration and to access other lattice components.
        """
        LatticeBase.__init__(self, lattice)

    @abstractmethod
    def __call__(self, f: np.ndarray):
        """
        """
        ...


class PipeNative(PipeBase):
    """
    """

    def __init__(self, lattice, pipes: [PipeBase]):
        """Trivial Constructor for Class PipeBase

        Parameters
        ----------
        lattice: Lattice
            The associated lattice object.
            Used for configuration and to access other lattice components.
        """
        PipeBase.__init__(self, lattice)

    def __call__(self, f: np.ndarray):
        """
        """


PIPE_STEP = 0


class Pipeline:
    """
    """

    pipes: [PipeBase]

    def __init__(self, pipes: [Union[PipeBase, int]]):
        """Trivial Constructor for Class PipeBase

        Parameters
        ----------
        pipes: [Union[PipeBase, int]]
            The Pipes that build up the Pipeline
        """
        self.pipes = pipes

    def __call__(self, f: np.ndarray, till_step: int, step: int = 0):
        while step < till_step:
            for pipe in self.pipes:
                if pipe == PIPE_STEP:
                    step = step + 1
                else:
                    pipe(f)
