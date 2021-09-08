from . import *


class NativeCollisionBGK(NativeCollision):
    _name = 'bgkCollision'

    def __init__(self, equilibrium: NativeEquilibrium = None, support_no_collision_mask=False):
        super().__init__(equilibrium, support_no_collision_mask)

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        return NativeCollisionBGK(equilibrium, support_no_collision_mask)

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
