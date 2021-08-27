import json

from lettuce.gen_native import *


class NativeStencil:
    """
    """

    name = 'invalidStencil'

    d_: int = None
    q_: int = None
    e_: [[int]] = None
    w_: [float] = None
    cs_: float = None

    @staticmethod
    def __init__():
        raise NotImplementedError("This class is not meant to be constructed "
                                  "as it provides only static fields and methods!")

    @classmethod
    def d(cls, gen: 'GeneratorKernel'):
        """
        """
        if not gen.registered('d'):
            gen.register('d')

            # generate
            gen.hrd(f"constexpr index_t d = {cls.d_};")

    @classmethod
    def q(cls, gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('q'):
            gen.register('q')

            # generate
            gen.hrd(f"constexpr index_t q = {cls.q_};")

    @classmethod
    def e(cls, gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('e'):
            gen.register('e')

            # requirements
            cls.d(gen)
            cls.q(gen)

            # generate
            buffer = [f"{{{', '.join([str(x) + '.0' for x in e])}}}" for e in cls.e_]
            gen.hrd(f"constexpr scalar_t e[q][d] = {{{', '.join(buffer)}}};")

    @classmethod
    def w(cls, gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('w'):
            gen.register('w')

            # requirements
            cls.q(gen)

            # generate
            buffer = [json.dumps(w) for w in cls.w_]
            buffer = ',\n                               '.join(buffer)
            gen.hrd(f"constexpr scalar_t w[q] = {{{buffer}}};")

    @classmethod
    def cs(cls, gen: 'GeneratorKernel'):
        """
        """

        if not gen.registered('cs'):
            gen.register('cs')

            # generate
            gen.hrd(f"constexpr scalar_t cs = {json.dumps(cls.cs_)};")
