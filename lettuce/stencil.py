import numpy as np

from . import *


class Stencil:
    e: np.ndarray
    w: np.ndarray
    cs: np.float
    opposite: [int]

    @classmethod
    def d(cls):
        return cls.e.shape[1]

    @classmethod
    def q(cls):
        return cls.e.shape[0]


class D1Q3(Stencil):
    native_class = native_generator.NativeD1Q3
    e = np.array([[0], [1], [-1]])
    w = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
    cs = 1 / np.sqrt(3)
    opposite = [0, 2, 1]


class D2Q9(Stencil):
    native_class = native_generator.NativeD2Q9
    e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
    w = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
    cs = 1 / np.sqrt(3)
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]


class D3Q19(Stencil):
    native_class = native_generator.NativeD3Q19
    e = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0]
    ])
    w = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12)
    cs = 1 / np.sqrt(3)
    opposite = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13,
        16, 15, 18, 17
    ]


class D3Q27(Stencil):
    native_class = native_generator.NativeD3Q27
    e = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 1, 1],
        [-1, -1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, 1, 1]
    ])
    w = np.array([8.0 / 27.0] + [2.0 / 27.0] * 6 + [1.0 / 54.0] * 12 + [1.0 / 216.0] * 8)
    cs = 1 / np.sqrt(3)
    opposite = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13,
        16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25
    ]
