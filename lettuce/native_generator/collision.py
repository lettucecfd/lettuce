from . import *


class NativeCollision(NativeLatticeBase):
    def collision(self, generator: 'GeneratorKernel'):
        raise AbstractMethodInvokedError()
