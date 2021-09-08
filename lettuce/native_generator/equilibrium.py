from . import *


class NativeEquilibrium(NativeLatticeBase):
    def f_eq(self, generator: 'GeneratorKernel'):
        raise AbstractMethodInvokedError()
