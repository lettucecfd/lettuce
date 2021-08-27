import math

from lettuce.gen_native import *


class NativeD3Q27(NativeStencil):
    name = 'd3q27'
    d_ = 3
    q_ = 27
    e_ = NativeD3Q19.e_ + [[1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1],
                           [1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, 1, 1]]
    w_ = [8. / 27.] + [2. / 27.] * 6 + [1. / 54.] * 12 + [1. / 216.] * 8
    cs_ = 1. / math.sqrt(3.)

    @staticmethod
    def __init__():
        super().__init__()
