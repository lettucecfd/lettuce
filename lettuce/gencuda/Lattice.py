from lettuce.gencuda import KernelGenerator


class Lattice:
    """
    """

    def __init__(self):
        """
        """
        pass

    def rho_inv(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('rho_inv'):
            gen.register('rho_inv')

            # dependencies
            gen.lattice.rho(gen)

            # generate
            gen.nde('const auto rho_inv = 1.0 / rho;')

    def rho(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('rho'):
            gen.register('rho')

            # generate
            f_eq_sum = ' + '.join([f"f_reg[{q}]" for q in range(gen.stencil.q_)])

            gen.nde(f"const auto rho = {f_eq_sum};")

    def u(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('u'):
            gen.register('u')

            # dependencies
            gen.stencil.d(gen)
            gen.stencil.e(gen)

            if gen.stencil.d_ > 1:
                gen.lattice.rho_inv(gen)

            # generate
            div_rho = ' * rho_inv' if gen.stencil.d_ > 1 else ' / rho'

            gen.nde(f"const scalar_t u[d]{{")
            for d in range(gen.stencil.d_):
                summands = []
                for q in range(gen.stencil.q_):
                    summands.append(f"e[{q}][{d}] * f_reg[{q}]")
                gen.nde(f"    ({' + '.join(summands)})" + div_rho + ',')
            gen.nde('};')
            gen.nde()
