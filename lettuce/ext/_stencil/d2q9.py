from ... import Stencil

__all__ = ['D2Q9']


class D2Q9(Stencil):
    def __init__(self):
        self.e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
        self.w = [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4
        self.opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
