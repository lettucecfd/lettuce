from . import *


class NativeEquilibriumQuadratic(NativeEquilibrium):
    """
    """

    name = 'quadraticEquilibrium'

    @staticmethod
    def __init__():
        super().__init__()

    @staticmethod
    def uxu(gen: 'GeneratorKernel'):
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

    @staticmethod
    def exu(gen: 'GeneratorKernel'):
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

    @staticmethod
    def cs_pow_two(gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('cs_pow_two<scalar_t>'):
            gen.register('cs_pow_two<scalar_t>')

            # dependencies
            gen.stencil.cs(gen)

            # generate
            gen.nde('constexpr auto cs_pow_two = cs * cs;')

    @classmethod
    def two_cs_pow_two(cls, gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('two_cs_pow_two<scalar_t>'):
            gen.register('two_cs_pow_two<scalar_t>')

            # dependencies
            cls.cs_pow_two(gen)

            # generate
            gen.nde('constexpr auto two_cs_pow_two = cs_pow_two + cs_pow_two;')

    @classmethod
    def f_eq_tmp(cls, gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('f_eq_tmp'):
            gen.register('f_eq_tmp')

            # dependencies
            cls.exu(gen)
            cls.cs_pow_two(gen)

            # generate
            gen.cln('const auto f_eq_tmp = exu / cs_pow_two;')

    @classmethod
    def f_eq(cls, gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('f_eq'):
            gen.register('f_eq')

            # dependencies
            gen.lattice.rho(gen)
            cls.exu(gen)
            cls.uxu(gen)
            cls.two_cs_pow_two(gen)
            cls.f_eq_tmp(gen)
            gen.stencil.w(gen)

            # generate
            gen.cln('const auto f_eq = '
                    'rho * (((exu + exu - uxu) / two_cs_pow_two) + (0.5 * (f_eq_tmp * f_eq_tmp)) + 1.0) * w[i];')
