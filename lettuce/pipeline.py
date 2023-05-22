from abc import abstractmethod
from typing import Union
import numpy as np

from . import PipelineStep


class Pipeline:
    """
    """

    pipeline_steps: [PipelineStep]

    def __init__(self, pipeline_steps: [(PipelineStep, int)]):
        """Trivial Constructor for Class PipeBase

        Parameters
        ----------
        pipes: [Union[PipeBase, int]]
            The Pipes that build up the Pipeline
        """
        self.pipeline_steps = pipeline_steps

    def __call__(self, f: np.ndarray, till_step: int, step: int = 0):
        while step < till_step:
            for pipe in self.pipeline_steps:
                pipe(f)
