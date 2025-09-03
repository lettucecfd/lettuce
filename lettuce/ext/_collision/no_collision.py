import torch

from ... import Flow, Collision
from ...cuda_native.ext import NativeNoCollision

__all__ = ['NoCollision']


class NoCollision(Collision):
    def __call__(self, flow: 'Flow') -> torch.Tensor:
        return flow.f

    def native_available(self) -> bool:
        return True

    def native_generator(self, index: int) -> 'NativeCollision':
        return NativeNoCollision(index)
