from ... import Stencil

__all__ = ['D1Q3']


class D1Q3(Stencil):
    def __init__(self):
        self.e = [[0], [1], [-1]]
        self.w = [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]
        self.opposite = [0, 2, 1]
