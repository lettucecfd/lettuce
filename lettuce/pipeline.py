from abc import abstractmethod
from typing import Union, Callable
import numpy as np

from . import PipelineStep


class Pipeline(PipelineStep):
    """
    """

    def native_available(self) -> bool:
        return True

    def create_native(self) -> ['NativeLatticeBase']:
        return

    pipeline_steps: [(PipelineStep, Union[int, Callable[[int], bool]])]

    def __init__(self, lattice, pipeline_steps):
        PipelineStep.__init__(self, lattice=lattice)
        self.pipeline_steps = pipeline_steps

    def __call__(self, f: np.ndarray, step=1):
        pipeline_steps = []
        for step, condition in self.pipeline_steps:
            if condition is int and step % condition == 0:
                pipeline_steps.append(step)
            elif condition(step):
                pipeline_steps.append(step)


