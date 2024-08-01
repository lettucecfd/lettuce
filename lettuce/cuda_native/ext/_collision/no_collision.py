from typing import Optional

from ... import NativeCollision

__all__ = ['NativeNoCollision']


class NativeNoCollision(NativeCollision):

    @staticmethod
    def create(force: Optional['NativeForce'] = None):
        assert force is None
        return NativeNoCollision()

    def generate(self, generator: 'Generator'):
        return
