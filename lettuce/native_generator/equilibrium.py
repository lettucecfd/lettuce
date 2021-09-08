from . import *


class NativeEquilibrium(NativeLatticeBase):
    def f_eq(self, generator: 'GeneratorKernel'):
        raise AbstractMethodInvokedError()


class NativeQuadraticEquilibrium(NativeEquilibrium):
    _name = 'Quadratic'

    def __init__(self):
        super().__init__()

    def uxu(self, generator: 'GeneratorKernel'):
        if not generator.registered('uxu'):
            generator.register('uxu')

            # dependencies
            generator.lattice.u(generator)

            # generate
            summands = []
            for d in range(generator.stencil.stencil.d()):
                summands.append(f"u[{d}] * u[{d}]")

            generator.nde(f"const auto uxu = {' + '.join(summands)};")

    def exu(self, generator: 'GeneratorKernel'):
        if not generator.registered('exu'):
            generator.register('exu')

            # dependencies
            generator.stencil.e(generator)
            generator.lattice.u(generator)

            # generate
            summands = []
            for d in range(generator.stencil.stencil.d()):
                summands.append(f"e[i][{d}] * u[{d}]")

            generator.cln(f"const auto exu = {' + '.join(summands)};")

    def cs_pow_two(self, generator: 'GeneratorKernel'):
        if not generator.registered('cs_pow_two<scalar_t>'):
            generator.register('cs_pow_two<scalar_t>')

            # dependencies
            generator.stencil.cs(generator)

            # generate
            generator.nde('constexpr auto cs_pow_two = cs * cs;')

    def two_cs_pow_two(self, generator: 'GeneratorKernel'):
        if not generator.registered('two_cs_pow_two<scalar_t>'):
            generator.register('two_cs_pow_two<scalar_t>')

            # dependencies
            self.cs_pow_two(generator)

            # generate
            generator.nde('constexpr auto two_cs_pow_two = cs_pow_two + cs_pow_two;')

    def f_eq_tmp(self, generator: 'GeneratorKernel'):
        if not generator.registered('f_eq_tmp'):
            generator.register('f_eq_tmp')

            # dependencies
            self.exu(generator)
            self.cs_pow_two(generator)

            # generate
            generator.cln('const auto f_eq_tmp = exu / cs_pow_two;')

    def f_eq(self, generator: 'GeneratorKernel'):
        if not generator.registered('f_eq'):
            generator.register('f_eq')

            # dependencies
            generator.lattice.rho(generator)
            self.exu(generator)
            self.uxu(generator)
            self.two_cs_pow_two(generator)
            self.f_eq_tmp(generator)
            generator.stencil.w(generator)

            # generate
            generator.cln('const auto f_eq = '
                          'rho * (((exu + exu - uxu) / two_cs_pow_two) + (0.5 * (f_eq_tmp * f_eq_tmp)) + 1.0) * w[i];')
