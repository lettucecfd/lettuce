import numpy as np

__all__ = ["Stencil", "D1Q3", "D2Q9", "D3Q15", "D3Q19", "D3Q27", "Symmetriesearch"]


class Stencil:

    @classmethod
    def D(cls):
        return cls.e.shape[1]

    @classmethod
    def Q(cls):
        return cls.e.shape[0]


class D1Q3(Stencil):
    e = np.array([[0], [1], [-1]])
    w = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
    cs = 1 / np.sqrt(3)
    opposite = [0, 2, 1]


class D2Q9(Stencil):
    e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
    w = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
    cs = 1 / np.sqrt(3)
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]


class D3Q15(Stencil):
    e = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 1],
        [-1, -1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, 1, 1]
    ])
    w = np.array([2.0 / 9.0] + [1.0 / 9.0] * 6 + [1.0 / 72.0] * 8)
    cs = 1 / np.sqrt(3)
    opposite = [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13
    ]


class D3Q19(Stencil):
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
    switch_xz = np.array([[4,3],[10,7],[8,9],[17,15],[16,18]])
    switch_yz = np.array([[2,1],[14,11],[12,13],[18,15],[16,17]])
    switch_xy = np.array([[6,5],[9,7],[8,10],[13,11],[12,14]])
    switch_x = np.array([[6,5],[8,7],[9,10],[13,11],[12,14]])
    switch_rotyx = np.array([[1,4],[15,17],[17,16],[11,10],[13,8]])
    switch_rotxy = np.array([[3,2],[15,18],[18,16],[7,14],[9,12]])
    switch_diagonal = np.array([[13,7],[12,10],[9,11],[8,14],[6,15]])
    switch_diagonal2 = np.array([[6,5],[13,7],[12,10],[9,11],[8,14]])
class D3Q27(Stencil):
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
class Symmetriesearch:
    def __init__(self, e):
        self.e = e

        self.s_a = np.array([[0, -1, -1, 0, 1, 1, 1, 2],
                        [1, -1, 1, 0, -1, 1, 1, 2],
                        [2, 1, 1, 0, 1, 1, -1, 2],
                        [2, -1, 1, 1, 1, 0, -1, 2],
                        [0, 1, -1, 1, 1, 0, 1, 2],
                        [1, 1, 1, 1, -1, 0, 1, 2]])

        self.switch_stencil_wall = []

        for side in range(6):
            self.opposite = []
            for i in range(len(e)):
                for j in range(len(e)):
                    if self.e[i, self.s_a[side, 0]] == self.s_a[side, 1] and self.e[i, 0] == self.s_a[side, 2] * self.e[j, self.s_a[side, 3]] and self.e[
                        i, 1] == self.s_a[side, 4] * self.e[j, self.s_a[side, 5]] and self.e[i, 2] == self.s_a[side, 6] * self.e[j, self.s_a[side, 7]]:
                        self.opposite.append((i, j))
            self.switch_stencil_wall.append(self.opposite)


        self.s_b = np.array([[0, -1, 1, 1, 0, -1, 1, 1, 2, 2],
                        [0, 1, 1, -1, 0, 1, 1, -1, 2, 2],
                        [0, 1, 1, 1, 0, -1, 1, -1, 2, 2],
                        [0, -1, 1, -1, 0, 1, 1, 1, 2, 2],
                        [0, -1, 2, 1, 0, 1, 2, -1, 1, 1],
                        [1, -1, 2, 1, 1, 1, 2, -1, 0, 0],
                        [1, 1, 2, 1, 0, -1, 2, -1, 0, 1],
                        [0, 1, 2, 1, 1, -1, 2, -1, 1, 0],
                        [1, 1, 2, -1, 1, -1, 2, 1, 0, 0],
                        [0, -1, 2, -1, 1, 1, 2, 1, 1, 0],
                        [1, -1, 2, -1, 0, 1, 2, 1, 0, 1],
                        [0, 1, 2, -1, 0, -1, 2, 1, 1, 1]])

        self.change_array_borders = []

        for b in range(12):
            self.opposite = []
            for i in range(len(e)):
                for j in range(len(e)):
                    if self.e[i, self.s_b[b, 0]] == self.s_b[b, 1] and self.e[i, self.s_b[b, 2]] == self.s_b[b, 3] and self.e[j, self.s_b[b, 4]] == self.s_b[
                        b, 5] and self.e[j, self.s_b[b, 6]] == self.s_b[b, 7] and self.e[i, self.s_b[b, 8]] == self.e[j, self.s_b[b, 9]]:
                        self.opposite.append((i, j))
            self.change_array_borders.append(self.opposite)

        self.opposite = []
        self.switch_stencil_corner = []

        for i in range(len(e)):
            for j in range(len(e)):
                if self.e[i, 0] != 0 and self.e[i, 1] != 0 and self.e[i, 2] != 0 and self.e[i, 0] == -self.e[j, 0] and self.e[i, 1] == -self.e[j, 1] and self.e[
                    i, 2] == -self.e[j, 2]:
                    self.opposite.append((i, j))
        self.switch_stencil_corner.append(self.opposite)


 
 
 
