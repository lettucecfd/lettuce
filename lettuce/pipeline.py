from abc import abstractmethod
from itertools import chain
from typing import Union, Callable
import numpy as np

from . import PipelineStep, Reporter, NativeWrite, LettuceBase, Generator, Simulation, NativeLettuceBase, NativeStencil
from .streaming import Read, Write, Streaming


class Pipeline(PipelineStep):
    """
    """

    pipeline_steps: [(PipelineStep, Union[int, Callable[[int], bool]])]

    def native_available(self) -> bool:
        return True

    def create_native(self) -> ['NativeLatticeBase']:
        return [self]  # creative is redirected to "on time" (__call__)

    def __init__(self, lattice, pipeline_steps):
        PipelineStep.__init__(self, lattice=lattice)
        self.pipeline_steps = pipeline_steps

    def __call__(self, simulation: 'Simulation'):

        pipeline_steps = []
        for step, condition in self.pipeline_steps:
            if isinstance(condition, int):
                if simulation.i % condition == 0:
                    pipeline_steps.append(step)
            elif condition(simulation.i):
                pipeline_steps.append(step)

        expect_read = True
        report_possible = True

        expect_write = False
        pipeline_step_possible = False

        # extract segments
        segment_steps = []
        segments = []
        for step in pipeline_steps:

            if isinstance(step, Read):
                if not expect_read:
                    raise RuntimeError("Invalid Pipeline")
                segment_steps.append(step)

                expect_read = False
                report_possible = False
                expect_write = True
                pipeline_step_possible = True
                continue

            if isinstance(step, Write):
                if not expect_write:
                    raise RuntimeError("Invalid Pipeline")
                segment_steps.append(step)

                segments.append(segment_steps.copy())
                segment_steps.clear()

                expect_read = True
                report_possible = True
                expect_write = False
                pipeline_step_possible = False
                continue

            if isinstance(step, PipelineStep):
                if not pipeline_step_possible:
                    raise RuntimeError("Invalid Pipeline")
                segment_steps.append(step)
                continue

            if isinstance(step, Reporter):
                if not report_possible:
                    raise RuntimeError("Invalid Pipeline")
                segments.append([step])
                continue

            raise RuntimeError("Invalid Pipeline Step!")

        segments_ = []
        for i in range(len(segments)):

            native_available = True
            for step in segments[i]:
                if not step.native_available():
                    native_available = False

            if native_available:
                stencil = NativeStencil(simulation.lattice.stencil)
                read, = segments[i][0].create_native()
                write, = segments[i][-1].create_native()
                pipeline_steps = [it.create_native() for it in segments[i][1:-1]]
                pipeline_steps = list(chain.from_iterable(pipeline_steps))

                generator = Generator(stencil, read, write, pipeline_steps)

                segment = generator.resolve()
                if segment is None:

                    buffer = generator.generate()
                    directory = generator.format(buffer)
                    generator.install(directory)

                    segment = generator.resolve()
                    if segment is None:
                        raise RuntimeError("Failed to install native Extension!")

                segments_.append(segment)

            else:
                segments_ += segments[i]

        segments = segments_

        for segment in segments:
            if isinstance(segment, Reporter):
                segment(simulation.i, simulation.flow.units.convert_time_to_pu(simulation.i), simulation.f)
                continue

            if isinstance(segment, LettuceBase):
                segment(simulation.f)
                continue

            segment(simulation)
        simulation.i += 1
