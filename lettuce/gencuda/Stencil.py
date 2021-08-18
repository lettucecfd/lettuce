import math

from lettuce.gencuda import KernelGenerator


class Stencil:
    """
    """

    def __init__(self, name: str, d: int, q: int, e: [[int]], w: [float], cs: float, opposite: [int]):
        """
        """

        self.name = name
        self.d_: int = d
        self.q_: int = q
        self.e_: [[int]] = e
        self.w_: [float] = w
        self.cs_: float = cs
        self.opposite_: [int] = opposite

    def d(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('d'):
            gen.register('d')

            # generate
            gen.hrd(f"constexpr index_t d = {self.d_};")

    def q(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('q'):
            gen.register('q')

            # generate
            gen.hrd(f"constexpr index_t q = {self.q_};")

    def e(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('e'):
            gen.register('e')

            # requirements
            self.d(gen)
            self.q(gen)

            # generate
            buffer = [f"{{{', '.join([str(x) + '.0' for x in e])}}}" for e in self.e_]
            gen.hrd(f"constexpr scalar_t e[q][d] = {{{', '.join(buffer)}}};")

    def w(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('w'):
            gen.register('w')

            # requirements
            self.q(gen)

            # generate
            buffer = [format(w, '.60g') for w in self.w_]
            buffer = ',\n                               '.join(buffer)
            gen.hrd(f"constexpr scalar_t w[q] = {{{buffer}}};")

    def cs(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('cs'):
            gen.register('cs')

            # generate
            gen.hrd(f"constexpr scalar_t cs = {format(self.cs_, '.60g')};")

    def opposite(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('opposite'):
            gen.register('opposite')

            # requirements
            self.q(gen)

            # generate
            buffer = [str(it) + 'u' for it in self.opposite_]
            gen.hrd(f"constexpr index_t opposite[q] = {{{', '.join(buffer)}}};")


class D1Q3(Stencil):
    def __init__(self):
        super().__init__(
            name='d1q3',
            d=1,
            q=3,
            e=[[0], [1], [-1]],
            w=[2. / 3., 1. / 6., 1. / 6.],
            cs=1. / math.sqrt(3.),
            opposite=[0, 2, 1])


class D2Q9(Stencil):
    def __init__(self):
        super().__init__(
            name='d2q9',
            d=2,
            q=9,
            e=[[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
            w=[4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4,
            cs=1. / math.sqrt(3.),
            opposite=[0, 3, 4, 1, 2, 7, 8, 5, 6])


class D3Q19(Stencil):
    def __init__(self):
        super().__init__(
            name='d3q19',
            d=3,
            q=19,
            e=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
               [0, 0, 1], [0, 0, -1], [0, 1, 1], [0, -1, -1], [0, 1, -1],
               [0, -1, 1], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
               [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]],
            w=[1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12,
            cs=1. / math.sqrt(3.),
            opposite=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])


class D3Q27(Stencil):
    def __init__(self):
        super().__init__(
            name='d3q27',
            d=3,
            q=27,
            e=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
               [0, 0, -1], [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1], [1, 0, 1],
               [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 1, 0], [-1, -1, 0], [1, -1, 0],
               [-1, 1, 0], [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1],
               [-1, 1, -1], [1, -1, -1], [-1, 1, 1]],
            w=[8. / 27.] + [2. / 27.] * 6 + [1. / 54.] * 12 + [1. / 216.] * 8,
            cs=1. / math.sqrt(3.),
            opposite=[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25])
