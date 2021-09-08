from . import *


class NativeStreamingNo(NativeStreaming):
    """
    """

    name = 'noStream'

    @staticmethod
    def __init__():
        super().__init__()

    @classmethod
    def read_write(cls, gen: 'GeneratorKernel', support_no_stream: bool, support_no_collision: bool):
        """
        """
        if not gen.registered('read_write()'):
            gen.register('read_write()')

            # dependencies:

            if support_no_collision:
                cls.no_collision_mask(gen)

            gen.stencil.q(gen)
            gen.cuda.length(gen, gen.stencil.d_ - 1, hook_into_kernel=True)
            gen.cuda.offset(gen)

            # read
            length_index = gen.stencil.d_ - 1

            gen.idx()
            gen.idx('scalar_t f_reg[q];')
            gen.idx(cls.read_frame_.format(source='f', length_index=length_index))

            if support_no_collision:
                indices = ', '.join([f"index{d}" for d in range(gen.stencil.d_)])
                gen.idx(f"if(!no_collision_mask[{indices}])")
            gen.idx('{')

            gen.wrt('}')
            gen.wrt()
            gen.wrt(cls.write_frame_.format(target='f', length_index=length_index))
