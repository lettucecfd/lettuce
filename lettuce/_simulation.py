import torch
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional
from abc import ABC, abstractmethod

from . import *
from .native import NativeCollision, NativeBoundary, Generator

__all__ = ['Collision', 'Boundary', 'Reporter', 'Simulation']


class Collision(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow'):
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self) -> 'NativeCollision':
        ...


class Boundary(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow'):
        ...

    @abstractmethod
    def make_no_collision_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self, index: int) -> 'NativeBoundary':
        ...


class Reporter(ABC):
    interval: int

    def __init__(self, interval: int):
        self.interval = interval

    @abstractmethod
    def __call__(self, flow: 'Flow'):
        ...


class Simulation:
    flow: 'Flow'
    context: 'Context'
    collision: 'Collision'
    boundaries: List['Boundary']
    no_collision_mask: Optional[torch.Tensor]
    no_streaming_mask: Optional[torch.Tensor]
    reporter: List['Reporter']

    def __init__(self, flow: 'Flow', collision: 'Collision',
                 boundaries: List['Boundary'], reporter: List['Reporter']):
        self.flow = flow
        self.context = flow.context
        self.collision = collision
        self.boundaries = [None] + sorted(boundaries, key=lambda b: str(b))  # Prefix [None] to ensure that Boundary.index (i.e., mask index) matches self.boundaries index
        self.reporter = reporter

        # ==================================== #
        # initialise masks based on boundaries #
        # ==================================== #

        # if there are no boundaries
        # leave the masks uninitialised
        self.no_collision_mask = None
        self.no_streaming_mask = None

        # else initialise the masks
        # based on the boundaries masks
        if len(self.boundaries) > 1:

            self.no_collision_mask = self.context.zero_tensor(
                    flow.resolution, dtype=torch.uint8
                )
            self.no_streaming_mask = self.context.zero_tensor(
                    [flow.stencil.q, *flow.resolution], dtype=torch.uint8
                )

            for i, boundary in enumerate(self.boundaries[1:], start=1):
                ncm = boundary.make_no_collision_mask([it for it in
                                                       self.flow.f.shape[1:]],
                                                      context=self.context)
                if ncm is not None:
                    self.no_collision_mask[ncm] = i
                nsm = boundary.make_no_streaming_mask([it for it in
                                                       self.flow.f.shape],
                                                      context=self.context)
                if nsm is not None:
                    self.no_streaming_mask |= nsm

        # ============================== #
        # generate native implementation #
        # ============================== #

        def collide_and_stream(*_, **__):
            self._collide()
            self._stream()

        self._collide_and_stream = collide_and_stream

        if self.context.use_native:

            # check for availability of native for all components

            if (self.flow.equilibrium is not None and not
                    self.flow.equilibrium.native_available()):
                name = self.flow.equilibrium.__class__.__name__
                print(f"native was requested, but equilibrium '{name}' "
                      f"does not support native.")
            if not self.collision.native_available():
                name = self.collision.__class__.__name__
                print(f"native was requested, but _collision '{name}' "
                      f"does not support native.")
            for boundary in self.boundaries[1:]:
                if not boundary.native_available():
                    name = boundary.__class__.__name__
                    print(f"native was requested, but _boundary '{name}' "
                          f"does not support native.")

            # create native equivalents

            native_equilibrium = None
            if self.flow.equilibrium is not None:
                native_equilibrium = self.flow.equilibrium.native_generator()

            native_collision = self.collision.native_generator()

            native_boundaries = []
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                native_boundaries.append(boundary.native_generator(i))

            # begin generating native module from native components

            generator = Generator(self.flow.stencil, native_collision,
                                  native_boundaries, native_equilibrium)

            native_kernel = generator.resolve()
            if native_kernel is None:

                buffer = generator.generate()
                directory = generator.format(buffer)
                generator.install(directory)

                native_kernel = generator.resolve()
                if native_kernel is None:
                    print('Failed to install native Extension!')
                    return

            # redirect collide and stream to native kernel

            self._collide_and_stream = native_kernel

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
                self.flow.f[i] = torch.where(self.no_streaming_mask[i],
                                             self.flow.f[i], new_fi)
        return self.flow.f

    def _collide(self):
        if self.no_collision_mask is None:
            self.flow.f = self.collision(self.flow)
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                self.flow.f = boundary(self.flow)
        else:
            torch.where(torch.eq(self.no_collision_mask, 0),
                        self.collision(self.flow), self.flow.f,
                        out=self.flow.f)
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                torch.where(torch.eq(self.no_collision_mask, i),
                            boundary(self.flow), self.flow.f, out=self.flow.f)
        return self.flow.f

    def _report(self):
        for reporter in self.reporter:
            reporter(self.flow)

    def __call__(self, num_steps):
        beg = timer()

        if self.flow.i == 0:
            for reporter in self.reporter:
                reporter(self.flow)

        for _ in range(num_steps):
            self._collide_and_stream(self)
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)
