from typing import Optional

from ... import NativeCollision

__all__ = ['NativeNoCollision']


class NativeNoCollision(NativeCollision):

    @staticmethod
    def create(index: int, force: None = None):
        assert force is None
        return NativeNoCollision(index)

    def generate(self, _):
        return
