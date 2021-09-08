from . import *


class NativeStreamingNo(NativeStreaming):
    _name = 'noStreaming'

    def __init__(self):
        super().__init__(False)

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeStreamingNo()

    def read_write(self, generator: 'GeneratorKernel'):
        if not generator.registered('read_write()'):
            generator.register('read_write()')

            # dependencies:

            generator.stencil.q(generator)
            generator.cuda.length(generator, generator.stencil.stencil.d() - 1, hook_into_kernel=True)
            generator.cuda.offset(generator)

            # read
            length_index = generator.stencil.stencil.d() - 1

            generator.idx()
            generator.idx('scalar_t f_reg[q];')
            generator.idx(self.read_frame_.format(source='f', length_index=length_index))

            generator.wrt()
            generator.wrt(self.write_frame_.format(target='f', length_index=length_index))
