from . import *


class NativeEquilibrium:
    """
    """

    name = 'invalidEquilibrium'

    @staticmethod
    def __init__():
        raise NotImplementedError("This class is not meant to be constructed "
                                  "as it provides only static fields and methods!")

    @staticmethod
    def f_eq(gen: 'GeneratorKernel'):
        raise NotImplementedError("This method is only implemented by concrete subclasses!")
