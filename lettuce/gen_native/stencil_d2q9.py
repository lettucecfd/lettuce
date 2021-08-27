import math

from lettuce.gen_native import *


class NativeD2Q9(NativeStencil):
    name = 'd2q9'
    d_ = 2
    q_ = 9
    e_ = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    w_ = [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4
    cs_ = 1. / math.sqrt(3.)

    @staticmethod
    def __init__():
        super().__init__()
