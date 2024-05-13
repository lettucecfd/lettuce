from lettuce.native import NativeEquilibrium

__all__ = ['NativeQuadraticEquilibrium']


class NativeQuadraticEquilibrium(NativeEquilibrium):

    # noinspection PyMethodMayBeStatic
    def generate_uxu(self, generator: 'Generator'):
        if not generator.registered('uxu'):
            generator.register('uxu')

            global_buf = generator.append_global_buffer
            global_buf('                    ')
            global_buf('  const auto uxu =  ')
            global_buf('      u[0] * u[0]   ')
            global_buf('    + u[1] * u[1]   ', cond=generator.stencil.d > 1)
            global_buf('    + u[2] * u[2]   ', cond=generator.stencil.d > 2)
            global_buf('  ;                 ')
            global_buf('                    ')

    # noinspection PyMethodMayBeStatic
    def generate_exu(self, generator: 'Generator'):
        if not generator.registered('exu'):
            generator.register('exu')

            global_buf = generator.append_global_buffer
            global_buf('  scalar_t exu[q];                ')
            global_buf('  # pragma unroll                 ')
            global_buf('  for (index_t i = 0; i < q; ++i) ')
            global_buf('  {                               ')
            global_buf('    exu[i] =                      ')
            global_buf('        e[i][0] * u[0]            ')
            global_buf('      + e[i][1] * u[1]            ', cond=generator.stencil.d > 1)
            global_buf('      + e[i][2] * u[2]            ', cond=generator.stencil.d > 2)
            global_buf('    ;                             ')
            global_buf('  }                               ')

    # noinspection PyMethodMayBeStatic
    def generate_cs_pow_two(self, generator: 'Generator'):
        if not generator.registered('cs_pow_two'):
            generator.register('cs_pow_two')

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
            self.generate_exu(generator)
            self.generate_uxu(generator)
            self.generate_cs_pow_two(generator)
            self.generate_two_cs_pow_two(generator)

            # generate
            generator.append_global_buffer('  scalar_t f_eq[q];                                                                                       ')
            generator.append_global_buffer('# pragma unroll                                                                                           ')
            generator.append_global_buffer('  for (index_t i = 0; i < q; ++i)                                                                         ')
            generator.append_global_buffer('  {                                                                                                       ')
            generator.append_global_buffer('    scalar_t f_eq_tmp = exu[i] / cs_pow_two;                                                              ')
            generator.append_global_buffer('    f_eq[i] = rho * w[i] * ((exu[i] + exu[i] - uxu) / two_cs_pow_two + 0.5 * f_eq_tmp * f_eq_tmp + 1.0);  ')
            generator.append_global_buffer('  }                                                                                                       ')
