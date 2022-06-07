from . import *


class NativeLattice:
    def generate_rho(self, generator: 'Generator'):
        if not generator.registered('rho'):
            generator.register('rho')

            q = generator.stencil.stencil.Q()

            # generate
            f_eq_sum = ' + '.join([f"f_reg[{i}]" for i in range(q)])

            generator.append_node_buffer(f"const auto rho = {f_eq_sum};")

    def generate_rho_inv(self, generator: 'Generator'):
        if not generator.registered('rho_inv'):
            generator.register('rho_inv')

            # dependencies
            self.generate_rho(generator)

            # generate
            generator.append_node_buffer('const auto rho_inv = 1.0 / rho;')

    def generate_u(self, generator: 'Generator'):
        if not generator.registered('u'):
            generator.register('u')

            # dependencies
            generator.stencil.generate_d(generator)
            generator.stencil.generate_e(generator)

            if generator.stencil.stencil.D() > 1:
                self.generate_rho_inv(generator)

            # generate
            div_rho = ' * rho_inv' if generator.stencil.stencil.D() > 1 else ' / rho'

            generator.append_node_buffer('const scalar_t u[d] = {')
            for i in range(generator.stencil.stencil.D()):
                summands = []
                for j in range(generator.stencil.stencil.Q()):
                    summands.append(f"e[{j}][{i}] * f_reg[{j}]")
                generator.append_node_buffer(f"    ({' + '.join(summands)})" + div_rho + ',')
            generator.append_node_buffer('};')
            generator.append_node_buffer()
