from lettuce.gen_native import *


class NativeCollisionNo(NativeCollision):
    """
    """

    name = 'noCollision'

    @staticmethod
    def __init__():
        super().__init__()

    @staticmethod
    def collide(gen: 'GeneratorKernel'):
        if not gen.registered('collide()'):
            gen.register('collide()')
