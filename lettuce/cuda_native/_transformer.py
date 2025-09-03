from abc import abstractmethod, ABC
from typing import Optional, List

from . import *

__all__ = [
    'NativeTransformer',
    'NativeBoundary',
    'NativeEquilibrium',
    'NativeCollision',
]


class NativeTransformer(ABC):
    index: int

    def __init__(self, index: int):
        self.index = index

    @staticmethod
    @abstractmethod
    def create(index: int):
        ...

    @abstractmethod
    def generate(self, reg: 'Registry'):
        ...


class NativeBoundary(NativeTransformer, ABC):
    ...


class NativeEquilibrium(ABC):
    @abstractmethod
    def f_eq(self, reg: 'Registry', q: int, rho: Optional[str] = None, u: Optional[List[str]] = None):
        ...


class NativeCollision(NativeTransformer, ABC):
    ...
