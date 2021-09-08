from . import *


class NativeCollisionNo(NativeCollision):
    _name = 'noCollision'

    def __init__(self):
        super().__init__(None, False)

    @property
    def name(self):
        return self._name

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        return NativeCollisionBGK()

    def collision(self, generator: 'GeneratorKernel'):
        if not generator.registered('collision()'):
            generator.register('collision()')
