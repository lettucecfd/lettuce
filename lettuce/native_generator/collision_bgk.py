from . import *


class NativeCollisionBGK(NativeCollision):
    _name = 'bgkCollision'

    def __init__(self):
        super().__init__()

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
            self.tau_inv(generator)
            generator.equilibrium.f_eq(generator)

            # generate
            generator.cln('f_reg[i] = f_reg[i] - (tau_inv * (f_reg[i] - f_eq));')
