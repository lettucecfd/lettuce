from . import *
from abc import abstractmethod


class NativeEquilibrium(NativeLatticeBase):
    @abstractmethod
    def generate_f_eq(self, generator: 'Generator'):
        ...


class NativeQuadraticEquilibrium(NativeEquilibrium):
    @property
    def name(self) -> str:
        return 'QuadraticEquilibrium'

    # noinspection PyMethodMayBeStatic
    def generate_uxu(self, generator: 'Generator'):
        if not generator.registered('uxu'):
            generator.register('uxu')

            # dependencies
            generator.lattice.generate_u(generator)
            d = generator.stencil.stencil.D()

            # generate
            node_buf = generator.append_node_buffer
            node_buf('                    ')
            node_buf('  const auto uxu =  ')
            node_buf('      u[0] * u[0]   ')
            node_buf('    + u[1] * u[1]   ', cond=d > 1)
            node_buf('    + u[2] * u[2]   ', cond=d > 2)
            node_buf('  ;                 ')
            node_buf('                    ')

    # noinspection PyMethodMayBeStatic
    def generate_exu(self, generator: 'Generator'):
        if not generator.registered('exu'):
            generator.register('exu')

            # dependencies
            generator.stencil.generate_e(generator)
            generator.lattice.generate_u(generator)
            d = generator.stencil.stencil.D()

            # generate
            dist_buf = generator.append_distribution_buffer
            dist_buf('  const auto exu =    ')
            dist_buf('      e[i][0] * u[0]  ')
            dist_buf('    + e[i][1] * u[1]  ', cond=d > 1)
            dist_buf('    + e[i][2] * u[2]  ', cond=d > 2)
            dist_buf('  ;                   ')

    # noinspection PyMethodMayBeStatic
    def generate_cs_pow_two(self, generator: 'Generator'):
        if not generator.registered('cs_pow_two'):
            generator.register('cs_pow_two')

            # dependencies
            generator.stencil.generate_cs(generator)

            # generate
            generator.append_node_buffer('constexpr auto cs_pow_two = cs * cs;')

    def generate_two_cs_pow_two(self, generator: 'Generator'):
        if not generator.registered('two_cs_pow_two<scalar_t>'):
            generator.register('two_cs_pow_two<scalar_t>')

            # dependencies
            self.generate_cs_pow_two(generator)

            # generate
            generator.append_node_buffer('constexpr auto two_cs_pow_two = cs_pow_two + cs_pow_two;')

    def generate_f_eq_tmp(self, generator: 'Generator'):
        if not generator.registered('f_eq_tmp'):
            generator.register('f_eq_tmp')

            # dependencies
            self.generate_exu(generator)
            self.generate_cs_pow_two(generator)

            # generate
            generator.append_distribution_buffer('const auto f_eq_tmp = exu / cs_pow_two;')

    def generate_f_eq(self, generator: 'Generator'):
        if not generator.registered('f_eq'):
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
