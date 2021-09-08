from typing import Optional

from . import *


class NativeCollision(NativeLatticeBase):
    _name = 'invalidCollision'

    equilibrium: Optional[NativeEquilibrium]
    support_no_collision_mask: bool

    # noinspection PyShadowingNames
    def __init__(self, equilibrium: NativeEquilibrium = None, support_no_collision_mask=False):
        self.equilibrium = equilibrium if equilibrium is not None else NativeEquilibriumQuadratic()
        self.support_no_collision_mask = support_no_collision_mask

    @property
    def name(self):
        equilibrium_name = self.equilibrium.name if equilibrium is not None else ''
        mask_name = 'Masked' if self.support_no_collision_mask else ''
        return f"{self._name}{equilibrium_name}{mask_name}"

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask):
        raise AbstractMethodInvokedError()

    def no_collision_mask(self, generator: 'GeneratorKernel'):
        if not generator.wrapper_hooked('no_collision_mask'):
            generator.pyr("assert hasattr(simulation, 'no_collision_mask')")
            generator.wrapper_hook('no_collision_mask', 'const at::Tensor no_collision_mask',
                                   'no_collision_mask', 'simulation.no_collision_mask')
        if not generator.kernel_hooked('no_collision_mask'):
            generator.kernel_hook('no_collision_mask', 'const byte_t* no_collision_mask',
                                  'no_collision_mask.data<byte_t>()')

    def collision(self, generator: 'GeneratorKernel'):
        raise AbstractMethodInvokedError()
