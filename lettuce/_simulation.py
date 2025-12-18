import warnings

import torch
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional
from abc import ABC, abstractmethod

from . import *
from .cuda_native import NativeCollision, Generator, StreamingStrategy

# todo StreamingStrategy was aliased here but should, see StreamingStrategy for todo
__all__ = ['Collision', 'Reporter', 'Simulation', 'BreakableSimulation', 'StreamingStrategy']


class Collision(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow'):
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self, index: int) -> 'NativeCollision':
        ...


class Reporter(ABC):
    interval: int

    def __init__(self, interval: int):
        self.interval = interval

    @abstractmethod
    def __call__(self, simulation: 'Simulation'):
        ...


class Simulation:
    flow: 'Flow'
    context: 'Context'
    collision: 'Collision'
    # pre_boundaries: List['Boundary']
    # post_boundaries: List['Boundary']
    no_collision_mask: Optional[torch.Tensor]
    no_streaming_mask: Optional[torch.Tensor]
    reporter: List['Reporter']
    streaming_strategy: StreamingStrategy

    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter'], streaming_strategy=StreamingStrategy.POST_STREAMING):
        self.flow = flow
        self.flow.collision = collision
        self.context = flow.context
        self.collision = collision
        self.collision_index = len(flow.pre_boundaries)
        self.transformer = (flow.pre_boundaries or []) + [collision] + (flow.post_boundaries or [])
        self.reporter = reporter
        self.pre_boundaries = flow.pre_boundaries
        self.post_boundaries = flow.post_boundaries
        self.streaming_strategy = streaming_strategy

        # ==================================== #
        # initialise masks based on boundaries #
        # ==================================== #

        # if there are no boundaries
        # leave the masks uninitialised
        self.no_collision_mask = None
        self.no_streaming_mask = None

        # else initialise the masks
        # based on the boundaries masks

        if len(self.pre_boundaries) + len(self.post_boundaries) > 0:

            self.no_collision_mask = self.context.full_tensor(
                flow.resolution, self.collision_index, dtype=torch.uint8)
            self.no_streaming_mask = self.context.full_tensor(
                [flow.stencil.q, *flow.resolution], self.collision_index, dtype=torch.uint8)

            for i, boundary in enumerate(self.pre_boundaries):
                ncm = boundary.make_no_collision_mask(
                    [it for it in self.flow.f.shape[1:]], context=self.context)
                if ncm is not None:
                    self.no_collision_mask[ncm] = i
                nsm = boundary.make_no_streaming_mask(
                    [it for it in self.flow.f.shape], context=self.context)
                if nsm is not None:
                    self.no_streaming_mask |= nsm

            for i, boundary in enumerate(self.post_boundaries,start=self.collision_index+1):
                ncm = boundary.make_no_collision_mask(
                    [it for it in self.flow.f.shape[1:]], context=self.context)
                if ncm is not None:
                    self.no_collision_mask[ncm] = i
                nsm = boundary.make_no_streaming_mask(
                    [it for it in self.flow.f.shape], context=self.context)
                if nsm is not None:
                    self.no_streaming_mask |= nsm

        # =================================== #
        # generate cuda_native implementation #
        # =================================== #

        if streaming_strategy.pre_streaming() and streaming_strategy.post_streaming():
            def collide_and_stream(*_, **__):
                self._stream()
                self._collide()
                self._stream()
        elif streaming_strategy.post_streaming():
            def collide_and_stream(*_, **__):
                self._collide()
                self._stream()
        elif streaming_strategy.pre_streaming():
            def collide_and_stream(*_, **__):
                self._stream()
                self._collide()
        else:
            def collide_and_stream(*_, **__):
                self._collide()

        self._collide_and_stream = collide_and_stream

        if self.context.use_native:

            # check for availability of cuda_native for all components

            if (self.flow.equilibrium is not None
                    and not self.flow.equilibrium.native_available()):
                name = self.flow.equilibrium.__class__.__name__
                print(f"cuda_native was requested, but equilibrium '{name}' "
                      f"does not support cuda_native.")
            if not self.collision.native_available():
                name = self.collision.__class__.__name__
                print(f"cuda_native was requested, but collision '{name}' "
                      f"does not support cuda_native.")
            for boundary in self.pre_boundaries + self.post_boundaries:
                if not boundary.native_available():
                    name = boundary.__class__.__name__
                    print(f"cuda_native was requested, but boundary '{name}' "
                          f"does not support cuda_native.")

            # create cuda_native equivalents

            native_equilibrium = None
            if self.flow.equilibrium is not None:
                native_equilibrium = self.flow.equilibrium.native_generator()

            native_collision = self.collision.native_generator(self.collision_index)

            native_pre_boundaries = []
            for i, boundary in enumerate(self.pre_boundaries):
                native_pre_boundaries.append(boundary.native_generator(i))

            native_post_boundaries = []
            for i, boundary in enumerate(self.post_boundaries, start=self.collision_index+1):
                native_post_boundaries.append(boundary.native_generator(i))

            # begin generating cuda_native module from cuda_native components

            generator = Generator(self.flow.stencil,
                                  collision = native_collision,
                                  pre_boundaries = native_pre_boundaries,
                                  post_boundaries = native_post_boundaries,
                                  equilibrium = native_equilibrium,
                                  streaming_strategy = streaming_strategy)
            native_kernel = generator.resolve()
            if native_kernel is None:

                buffer = generator.generate()
                directory = generator.format(buffer)
                generator.install(directory)

                native_kernel = generator.resolve()
                if native_kernel is None:
                    print('Failed to install cuda_native Extension!')
                    return

            # redirect collide and stream to cuda_native kernel

            self._collide_and_stream = native_kernel

    def step(self, num_steps: int):
        warnings.warn("lt.Simulation.step() is deprecated and will be "
                      "removed in a future version. Instead, call simulation "
                      "directly: simulation(num_steps)", DeprecationWarning)
        return self(num_steps)

    @property
    def units(self):
        return self.flow.units

    @staticmethod
    def __stream(f, i, e, d):
        return torch.roll(f[i], shifts=tuple(e[i]), dims=tuple(np.arange(d)))

    def _stream(self):
        for i in range(1, self.flow.stencil.q):
            if self.no_streaming_mask is None:
                self.flow.f[i] = self.__stream(self.flow.f, i,
                                               self.flow.stencil.e,
                                               self.flow.stencil.d)
            else:
                new_fi = self.__stream(self.flow.f, i, self.flow.stencil.e,
                                       self.flow.stencil.d)
                self.flow.f[i] = torch.where(torch.eq(
                    self.no_streaming_mask[i], 1), self.flow.f[i], new_fi)
        return self.flow.f

    def _collide(self):
        if self.no_collision_mask is None:
            for boundary in self.pre_boundaries:
                self.flow.f = boundary(self.flow)
            self.flow.f = self.collision(self.flow)
            for boundary in self.post_boundaries:
                self.flow.f = boundary(self.flow)
        else:
            for i, boundary in enumerate(self.pre_boundaries):
                torch.where(torch.eq(self.no_collision_mask, i),
                            boundary(self.flow), self.flow.f, out=self.flow.f)
            torch.where(torch.eq(self.no_collision_mask, self.collision_index),
                        self.collision(self.flow), self.flow.f,
                        out=self.flow.f)
            for i, boundary in enumerate(self.post_boundaries, start=self.collision_index+1):
                torch.where(torch.eq(self.no_collision_mask, i),
                            boundary(self.flow), self.flow.f, out=self.flow.f)
        return self.flow.f

    def _report(self):
        for reporter in self.reporter:
            reporter(self)

    def __call__(self, num_steps):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        for _ in range(num_steps):
            self._collide_and_stream(self)
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)

class BreakableSimulation(Simulation):
    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter']):
        super().__init__(flow, collision, reporter)

    def __call__(self, num_steps: int):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        while self.flow.i < num_steps:
            # flow.i is used instead of _,
            # because it is mutable by the reporter!
            self._collide_and_stream(self)
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)