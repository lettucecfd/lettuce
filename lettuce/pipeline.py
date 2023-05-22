from abc import abstractmethod
from typing import Union
import numpy as np

from . import LettuceBase


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
