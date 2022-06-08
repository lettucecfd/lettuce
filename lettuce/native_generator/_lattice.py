from . import *


class NativeLattice:
    def get_lattice_coordinate(self, generator: 'Generator', coordinate):
        generator.cuda.generate_index(generator)
        generator.cuda.generate_dimension(generator)

        if len(coordinate) < 2 or len(coordinate) > 4:
            raise RuntimeError("Invalid coordinate passed! Faulty Generator implementation.")

        q, x = coordinate[0], coordinate[1]
        if len(coordinate) == 2:
            return f"{q} * dimension[0] + {x}"

        y = coordinate[2]
        if len(coordinate) == 3:
            return f"({q} * dimension[0] + {x}) * dimension[1]  + {y}"

        z = coordinate[3]
        return f"(({q} * dimension[0] + {x}) * dimension[1] + {y}) * dimension[2] + {z}"

    def get_mask_coordinate(self, generator: 'Generator', coordinate):
        generator.cuda.generate_index(generator)
        generator.cuda.generate_dimension(generator)

        if len(coordinate) < 1 or len(coordinate) > 3:
            raise RuntimeError("Invalid coordinate passed! Faulty Generator implementation.")

        x = coordinate[0]
        if len(coordinate) == 2:
            return f"{x}"

        y = coordinate[1]
        if len(coordinate) == 3:
            return f"{x} * dimension[0]  + {y}"

        z = coordinate[2]
        return f"({x} * dimension[0] + {y}) * dimension[1] + {z}"

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
