from typing import Optional

from . import *


class NativeCollision(NativeLatticeBase):
    _name = 'invalidCollision'

    equilibrium: Optional[NativeEquilibrium]
    support_no_collision_mask: bool

    # noinspection PyShadowingNames
    def __init__(self, equilibrium: NativeEquilibrium = None, support_no_collision_mask=False):
        self.equilibrium = equilibrium if equilibrium is not None else NativeQuadraticEquilibrium()
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


class NativeNoCollision(NativeCollision):
    _name = 'noCollision'

    def __init__(self):
        super().__init__(None, False)

    @property
    def name(self):
        return self._name

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        return NativeBGKCollision()

    def collision(self, generator: 'GeneratorKernel'):
        if not generator.registered('collision()'):
            generator.register('collision()')


class NativeBGKCollision(NativeCollision):
    _name = 'bgkCollision'

    def __init__(self, equilibrium: NativeEquilibrium = None, support_no_collision_mask=False):
        super().__init__(equilibrium, support_no_collision_mask)

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        return NativeBGKCollision(equilibrium, support_no_collision_mask)

    def tau_inv(self, generator: 'GeneratorKernel'):
        if not generator.wrapper_hooked('tau_inv'):
            generator.pyr("assert hasattr(simulation.collision, 'tau')")
            generator.wrapper_hook('tau_inv', 'const double tau_inv', 'tau_inv', '1./simulation.collision.tau')
        if not generator.kernel_hooked('tau_inv'):
            generator.kernel_hook('tau_inv', 'const scalar_t tau_inv', 'static_cast<scalar_t>(tau_inv)')

    def collision(self, generator: 'GeneratorKernel'):
        if not generator.registered('collide()'):
            generator.register('collide()')

            # dependencies

            if self.support_no_collision_mask:
                self.no_collision_mask(generator)

            self.tau_inv(generator)
            self.equilibrium.f_eq(generator)

            # generate
            if self.support_no_collision_mask:
                generator.idx(f"if(!no_collision_mask[offset])")

            generator.cln('f_reg[i] = f_reg[i] - (tau_inv * (f_reg[i] - f_eq));')
