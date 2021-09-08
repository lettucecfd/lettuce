from . import *


class NativeCollisionNo(NativeCollision):
    _name = 'noCollision'

    def __init__(self):
        super().__init__()

    def collision(self, generator: 'GeneratorKernel'):
        if not generator.registered('collide()'):
            generator.register('collide()')
