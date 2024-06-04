from lettuce.native import NativeEquilibrium

__all__ = ['NativeQuadraticEquilibrium']


class NativeQuadraticEquilibrium(NativeEquilibrium):

    # noinspection PyMethodMayBeStatic
    def generate_uxu(self, generator: 'Generator', u: str = None):
        if u is None or u == 'u':
            name = 'uxu'
            u = 'u'
        else:
            name = f"uxu_{u}"
        if not generator.registered(name):
            generator.register(name)

            global_buf = generator.append_global_buffer
            global_buf('                        ')
            global_buf(f"  const auto {name} =  ")
            global_buf(f"      {u}[0] * {u}[0]  ")
            global_buf(f"    + {u}[1] * {u}[1]  ", cond=generator.stencil.d > 1)
            global_buf(f"    + {u}[2] * {u}[2]  ", cond=generator.stencil.d > 2)
            global_buf('  ;                     ')
            global_buf('                        ')

    # noinspection PyMethodMayBeStatic
    def generate_exu(self, generator: 'Generator', u: str = None):
        if u is None or u == 'u':
            name = 'exu'
            u = 'u'
        else:
            name = f"exu_{u}"
        if not generator.registered(name):
            generator.register(name)

            global_buf = generator.append_global_buffer
            global_buf(f"  scalar_t {name}[q];            ")
            global_buf('  # pragma unroll                 ')
            global_buf('  for (index_t i = 0; i < q; ++i) ')
            global_buf('  {                               ')
            global_buf(f"    {name}[i] =                  ")
            global_buf(f"        e[i][0] * {u}[0]         ")
            global_buf(f"      + e[i][1] * {u}[1]         ", cond=generator.stencil.d > 1)
            global_buf(f"      + e[i][2] * {u}[2]         ", cond=generator.stencil.d > 2)
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

    def generate_f_eq(self, generator: 'Generator', rho: str = None, u: str = None):
        if (rho is None or rho == 'rho') and (u is None or u == 'u'):
            rho = 'rho'
            u = 'u'
            name = 'f_eq'
            uxu = 'uxu'
            exu = 'exu'
        else:
            rho = rho or "rho"
            u = u or "u"
            name = f"f_eq_{rho}_{u}"
            if u == 'u':
                uxu = 'uxu'
                exu = 'exu'
            else:
                uxu = f"uxu_{u}"
                exu = f"exu_{u}"
        if not generator.registered(name):
            generator.register(name)

            # dependencies
            self.generate_exu(generator, u)
            self.generate_uxu(generator, u)
            self.generate_cs_pow_two(generator)
            self.generate_two_cs_pow_two(generator)

            # generate
            generator.append_global_buffer(f"  scalar_t {name}[q];                                                                                               ")
            generator.append_global_buffer('# pragma unroll                                                                                                      ')
            generator.append_global_buffer('  for (index_t i = 0; i < q; ++i)                                                                                    ')
            generator.append_global_buffer('  {                                                                                                                  ')
            generator.append_global_buffer(f"    scalar_t f_eq_tmp = {exu}[i] / cs_pow_two;                                                                      ")
            generator.append_global_buffer(f"    {name}[i] = {rho} * w[i] * (({exu}[i] + {exu}[i] - {uxu}) / two_cs_pow_two + 0.5 * f_eq_tmp * f_eq_tmp + 1.0);  ")
            generator.append_global_buffer('  }                                                                                                                  ')
