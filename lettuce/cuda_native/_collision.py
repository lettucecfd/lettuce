from abc import ABC, abstractmethod
from typing import Optional

from . import *

__all__ = [
    'NativeEquilibrium',
    'NativeCollision',
]


class NativeEquilibrium(ABC):
    @abstractmethod
    def generate_f_eq(self, generator: 'Generator', rho: str = None, u: str = None):
        ...


class NativeCollision(ABC):

    def __init__(self):
        ABC.__init__(self)

    @staticmethod
    @abstractmethod
    def create(force: Optional['NativeForce'] = None):
        ...

    # noinspection PyMethodMayBeStatic
    def generate_no_collision_mask(self, generator: 'Generator'):
        if not generator.launcher_hooked('no_collision_mask'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation, 'no_collision_mask')")
            generator.launcher_hook('no_collision_mask', 'const at::Tensor no_collision_mask',
                                    'no_collision_mask', 'simulation.no_collision_mask')
        if not generator.kernel_hooked('no_collision_mask'):
            generator.kernel_hook('no_collision_mask', 'const byte_t* no_collision_mask',
                                  'no_collision_mask.data<byte_t>()')
