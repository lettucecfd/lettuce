import math

from . import *


class NativeD1Q3(NativeStencil):
    name = 'd1q3'
    d_ = 1
    q_ = 3
    e_ = [[0], [1], [-1]]
    w_ = [2. / 3., 1. / 6., 1. / 6.]
    cs_ = 1. / math.sqrt(3.)

    @staticmethod
    def __init__():
        super().__init__()
