from lettuce.gencuda import KernelGenerator


class Collision:
    """
    """

    name: str

    def __init__(self):
        """
        """
        self.name = 'invalid'

    def collide(self, gen: 'KernelGenerator'):
        """
        """
        assert False, 'Not implemented Error'


class BKGCollision(Collision):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.name = 'bgk'

    def tau_inv(self, gen: 'KernelGenerator'):
        """
        """
        if not gen.wrapper_hooked('tau_inv'):
            gen.py("assert hasattr(simulation.collision, 'tau')")
            gen.wrapper_hook('tau_inv', 'const double tau_inv', 'tau_inv', '1./simulation.collision.tau')
        if not gen.kernel_hooked('tau_inv'):
            gen.kernel_hook('tau_inv', 'const scalar_t tau_inv', 'static_cast<scalar_t>(tau_inv)')

    def collide(self, gen: 'KernelGenerator'):
        """
        """
        if not gen.registered('collide()'):
            gen.register('collide()')

            # dependencies
            self.tau_inv(gen)
            gen.equilibrium.f_eq(gen)

            # generate
            gen.cln('f_reg[i] = f_reg[i] - (tau_inv * (f_reg[i] - f_eq));')
