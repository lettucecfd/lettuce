from . import *


class NativeCollision:
    """
    """

    name = 'invalidCollision'

    @staticmethod
    def __init__():
        raise NotImplementedError("This class is not meant to be constructed "
                                  "as it provides only static fields and methods!")

    @staticmethod
    def collide(gen: 'GeneratorKernel'):
        raise NotImplementedError("This method is only implemented by concrete subclasses!")
