from . import *


class NativeEquilibrium(NativeLatticeBase):
    def generate_f_eq(self, generator: 'Generator'):
        raise NotImplementedError()


class NativeQuadraticEquilibrium(NativeEquilibrium):
    _name = 'Quad'

    def __init__(self):
        super().__init__()

    def generate_uxu(self, generator: 'Generator'):
        if generator.registered('uxu'):
            return

        generator.register('uxu')

        # dependencies
        generator.lattice.generate_u(generator)

        # generate
        summands = []
        for i in range(generator.stencil.stencil.D()):
            summands.append(f"u[{i}] * u[{i}]")

        generator.append_node_buffer(f"const auto uxu = {' + '.join(summands)};")

    def generate_exu(self, generator: 'Generator'):
        if generator.registered('exu'):
            return

        generator.register('exu')

        # dependencies
        generator.stencil.generate_e(generator)
        generator.lattice.generate_u(generator)

        # generate
        summands = []
        for i in range(generator.stencil.stencil.D()):
            summands.append(f"e[i][{i}] * u[{i}]")

        generator.append_distribution_buffer(f"const auto exu = {' + '.join(summands)};")

    def generate_cs_pow_two(self, generator: 'Generator'):
        if generator.registered('cs_pow_two<scalar_t>'):
            return

        generator.register('cs_pow_two<scalar_t>')

        # dependencies
        generator.stencil.generate_cs(generator)

        # generate
        generator.append_node_buffer('constexpr auto cs_pow_two = cs * cs;')

    def generate_two_cs_pow_two(self, generator: 'Generator'):
        if generator.registered('two_cs_pow_two<scalar_t>'):
            return

        generator.register('two_cs_pow_two<scalar_t>')

        # dependencies
        self.generate_cs_pow_two(generator)

        # generate
        generator.append_node_buffer('constexpr auto two_cs_pow_two = cs_pow_two + cs_pow_two;')

    def generate_f_eq_tmp(self, generator: 'Generator'):
        if generator.registered('f_eq_tmp'):
            return

        generator.register('f_eq_tmp')

        # dependencies
        self.generate_exu(generator)
        self.generate_cs_pow_two(generator)

        # generate
        generator.append_distribution_buffer('const auto f_eq_tmp = exu / cs_pow_two;')

    def generate_f_eq(self, generator: 'Generator'):
        if generator.registered('f_eq'):
            return

        generator.register('f_eq')

        # dependencies
        generator.lattice.generate_rho(generator)
        self.generate_exu(generator)
        self.generate_uxu(generator)
        self.generate_two_cs_pow_two(generator)
        self.generate_f_eq_tmp(generator)
        generator.stencil.generate_w(generator)

        # generate
        generator.append_distribution_buffer('const auto f_eq = '
                                             'rho * (((exu + exu - uxu) / two_cs_pow_two) + (0.5 * (f_eq_tmp * f_eq_tmp)) + 1.0) * w[i];')
