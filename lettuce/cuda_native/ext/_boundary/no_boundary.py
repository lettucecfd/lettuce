from ... import NativeBoundary

__all__ = ['NativeNoBoundary']


class NativeNoBoundary(NativeBoundary):
    def __init__(self, index):
        NativeBoundary.__init__(self, index)

    @staticmethod
    def create(index):
        return NativeNoBoundary(index)

    def generate(self, generator: 'Generator'):
        return
