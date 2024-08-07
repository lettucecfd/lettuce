from ... import Stencil

__all__ = ['D3Q15']


class D3Q15(Stencil):
    def __init__(self):
        self.e = [[0, 0, 0],
                  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                  [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, 1, 1]]
        self.w = [2.0 / 9.0] + [1.0 / 9.0] * 6 + [1.0 / 72.0] * 8
        self.opposite = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
