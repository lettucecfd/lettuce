from lettuce.gen_native import *


class NativeCollisionBGK(NativeCollision):
    """
    """

    name = 'bgkCollision'

    @staticmethod
    def __init__():
        super().__init__()

    @staticmethod
    def tau_inv(gen: 'GeneratorKernel'):
        """
        """
        if not gen.wrapper_hooked('tau_inv'):
            gen.pyr("assert hasattr(simulation.collision, 'tau')")
            gen.wrapper_hook('tau_inv', 'const double tau_inv', 'tau_inv', '1./simulation.collision.tau')
        if not gen.kernel_hooked('tau_inv'):
            gen.kernel_hook('tau_inv', 'const scalar_t tau_inv', 'static_cast<scalar_t>(tau_inv)')

    @classmethod
    def collide(cls, gen: 'GeneratorKernel'):
        """
        """
        if not gen.registered('collide()'):
            gen.register('collide()')

            # dependencies
            cls.tau_inv(gen)
            gen.equilibrium.f_eq(gen)

            # generate
            gen.cln('f_reg[i] = f_reg[i] - (tau_inv * (f_reg[i] - f_eq));')
