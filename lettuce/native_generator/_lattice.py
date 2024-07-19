from . import *


class NativeLattice:
    # noinspection PyMethodMayBeStatic
    def get_lattice_coordinate(self, generator: 'Generator', coord):
        generator.cuda.generate_index(generator)
        generator.cuda.generate_dimension(generator)

        d = generator.stencil.stencil.D() + 1
        assert d == len(coord), "Invalid Parameter passed"
        assert d > 1, "Method is undefined fot this Parameter"
        assert d <= 4, "Method is undefined fot this Parameter"

        if d == 2:
            return f"{coord[0]} * dimension[0] + {coord[1]}"

        if d == 3:
            return f"({coord[0]} * dimension[0] + {coord[1]}) * dimension[1]  + {coord[2]}"

        else:
            return f"(({coord[0]} * dimension[0] + {coord[1]}) * dimension[1] + {coord[2]}) * dimension[2] + {coord[3]}"

    # noinspection PyMethodMayBeStatic
    def get_mask_coordinate(self, generator: 'Generator', coord):
        generator.cuda.generate_index(generator)
        generator.cuda.generate_dimension(generator)

        d = generator.stencil.stencil.D()
        assert d == len(coord), "Invalid Parameter passed"
        assert d > 0, "Method is undefined fot this Parameter"
        assert d <= 3, "Method is undefined fot this Parameter"

        if d == 2:
            return f"{coord[0]}"

        if d == 3:
            return f"{coord[0]} * dimension[0]  + {coord[1]}"

        else:
            return f"({coord[0]} * dimension[0] + {coord[1]}) * dimension[1] + {coord[2]}"

    # noinspection PyMethodMayBeStatic
    def generate_rho(self, generator: 'Generator'):
        if not generator.registered('rho'):
            generator.register('rho')

            q = generator.stencil.stencil.Q()

            # generate
            f_reg_sum = ' + '.join([f"f_reg[{i}]" for i in range(q)])
            generator.append_node_buffer(f"const auto rho = {f_reg_sum};")

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

            q = generator.stencil.stencil.Q()
            d = generator.stencil.stencil.D()

            if d > 1:
                self.generate_rho_inv(generator)

            # generate
            div_rho = ' * rho_inv' if d > 1 else ' / rho'

            node_buf = generator.append_node_buffer

            node_buf('                           ')
            node_buf('  const scalar_t u[d] = {  ')

            for i in range(d):
                summands = [f"e[{j}][{i}] * f_reg[{j}]" for j in range(q)]
                node_buf(f"    ({' + '.join(summands)}) {div_rho},  ")

            node_buf('  };                       ')
            node_buf('                           ')
