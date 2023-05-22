from . import *
from abc import abstractmethod


class NativeEquilibrium(NativeLettuceBase):
    @abstractmethod
    def generate_f_eq(self, generator: 'Generator'):
        ...


class NoEquilibrium(NativeEquilibrium):
    @property
    def name(self) -> str:
        return 'NoEquilibrium'

    def generate_f_eq(self, generator: 'Generator'):
        raise NotImplementedError("Method is not expected to be called ever!")


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
            global_buf = generator.append_global_buffer
            global_buf('                    ')
            global_buf('  const auto uxu =  ')
            global_buf('      u[0] * u[0]   ')
            global_buf('    + u[1] * u[1]   ', cond=d > 1)
            global_buf('    + u[2] * u[2]   ', cond=d > 2)
            global_buf('  ;                 ')
            global_buf('                    ')

    # noinspection PyMethodMayBeStatic
    def generate_exu(self, generator: 'Generator'):
        if not generator.registered('exu'):
            generator.register('exu')

            # dependencies
            generator.stencil.generate_e(generator)
            generator.lattice.generate_u(generator)
            d = generator.stencil.stencil.D()

            # generate

            global_buf = generator.append_global_buffer
            global_buf('  scalar_t exu[q];                ')
            global_buf('  # pragma unroll                 ')
            global_buf('  for (index_t i = 0; i < q; ++i) ')
            global_buf('  {                               ')
            global_buf('    exu[i] =                      ')
            global_buf('        e[i][0] * u[0]            ')
            global_buf('      + e[i][1] * u[1]            ', cond=d > 1)
            global_buf('      + e[i][2] * u[2]            ', cond=d > 2)
            global_buf('    ;                             ')
            global_buf('  }                               ')

    # noinspection PyMethodMayBeStatic
    def generate_cs_pow_two(self, generator: 'Generator'):
        if not generator.registered('cs_pow_two'):
            generator.register('cs_pow_two')

            # dependencies
            generator.stencil.generate_cs(generator)

            # generate
            generator.append_global_buffer('constexpr auto cs_pow_two = cs * cs;')

    def generate_two_cs_pow_two(self, generator: 'Generator'):
        if not generator.registered('two_cs_pow_two<scalar_t>'):
            generator.register('two_cs_pow_two<scalar_t>')

            # dependencies
            self.generate_cs_pow_two(generator)

            # generate
            generator.append_global_buffer('constexpr auto two_cs_pow_two = cs_pow_two + cs_pow_two;')

    def generate_f_eq(self, generator: 'Generator'):
        if not generator.registered('f_eq'):
            generator.register('f_eq')

            # dependencies
            generator.lattice.generate_rho(generator)
            self.generate_exu(generator)
            self.generate_uxu(generator)
            self.generate_cs_pow_two(generator)
            self.generate_two_cs_pow_two(generator)
            generator.stencil.generate_w(generator)

            # generate
            global_buf = generator.append_global_buffer
            global_buf('  scalar_t f_eq[q];                                                                                                  ')
            global_buf('  # pragma unroll                                                                                                    ')
            global_buf('  for (index_t i = 0; i < q; ++i)                                                                                    ')
            global_buf('  {                                                                                                                  ')
            global_buf('    scalar_t f_eq_tmp = exu[i] / cs_pow_two;                                                                         ')
            global_buf('    f_eq[i] = rho * w[i] * ((exu[i] + exu[i] - uxu) / two_cs_pow_two + 0.5 * f_eq_tmp * f_eq_tmp + 1.0); ')
            global_buf('  }                                                                                                                  ')
