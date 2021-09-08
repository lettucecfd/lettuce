from . import *


class NativeStreamingNo(NativeStreaming):

    _name = 'noStreaming'

    def read_write(self, generator: 'GeneratorKernel', support_no_stream: bool, support_no_collision: bool):
        if not generator.registered('read_write()'):
            generator.register('read_write()')

            # dependencies:

            if support_no_collision:
                self.no_collision_mask(generator)

            generator.stencil.q(generator)
            generator.cuda.length(generator, generator.stencil.d_ - 1, hook_into_kernel=True)
            generator.cuda.offset(generator)

            # read
            length_index = generator.stencil.d_ - 1

            generator.idx()
            generator.idx('scalar_t f_reg[q];')
            generator.idx(self.read_frame_.format(source='f', length_index=length_index))

            if support_no_collision:
                indices = ', '.join([f"index{d}" for d in range(generator.stencil.d_)])
                generator.idx(f"if(!no_collision_mask[{indices}])")
            generator.idx('{')

            generator.wrt('}')
            generator.wrt()
            generator.wrt(self.write_frame_.format(target='f', length_index=length_index))
