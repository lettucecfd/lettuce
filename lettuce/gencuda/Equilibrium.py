from lettuce.gencuda import KernelGenerator


class Equilibrium:
    """
    """

    def __init__(self):
        """
        """
        self.name = 'invalid'

    def f_eq(self, gen: 'KernelGenerator'):
        """
        """
        pass


class QuadraticEquilibrium(Equilibrium):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.name = 'quadratic'

    def uxu(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('uxu'):
            gen.register('uxu')

            # dependencies
            gen.lattice.u(gen)

            # generate
            summands = []
            for d in range(gen.stencil.d_):
                summands.append(f"u[{d}] * u[{d}]")

            gen.nde(f"const auto uxu = {' + '.join(summands)};")

    def exu(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('exu'):
            gen.register('exu')

            # dependencies
            gen.stencil.e(gen)
            gen.lattice.u(gen)

            # generate
            summands = []
            for d in range(gen.stencil.d_):
                summands.append(f"e[i][{d}] * u[{d}]")

            gen.cln(f"const auto exu = {' + '.join(summands)};")

    def cs_pow_two(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('cs_pow_two<scalar_t>'):
            gen.register('cs_pow_two<scalar_t>')

            # dependencies
            gen.stencil.cs(gen)

            # generate
            gen.nde('constexpr auto cs_pow_two = cs * cs;')

    def two_cs_pow_two(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('two_cs_pow_two<scalar_t>'):
            gen.register('two_cs_pow_two<scalar_t>')

            # dependencies
            self.cs_pow_two(gen)

            # generate
            gen.nde('constexpr auto two_cs_pow_two = cs_pow_two + cs_pow_two;')

    def f_eq_tmp(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('f_eq_tmp'):
            gen.register('f_eq_tmp')

            # dependencies
            self.exu(gen)
            self.cs_pow_two(gen)

            # generate
            gen.cln('const auto f_eq_tmp = exu / cs_pow_two;')

    def f_eq(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('f_eq'):
            gen.register('f_eq')

            # dependencies
            gen.lattice.rho(gen)
            self.exu(gen)
            self.uxu(gen)
            self.two_cs_pow_two(gen)
            self.f_eq_tmp(gen)
            gen.stencil.w(gen)

            # generate
            gen.cln('const auto f_eq = '
                    'rho * (((exu + exu - uxu) / two_cs_pow_two) + (0.5 * (f_eq_tmp * f_eq_tmp)) + 1.0) * w[i];')
