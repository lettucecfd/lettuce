from . import *


class NativeStreaming(NativeLatticeBase):
    _name = 'invalid'

    support_no_streaming_mask: bool

    def __init__(self, support_no_streaming_mask=False):
        self.support_no_streaming_mask = support_no_streaming_mask

    @property
    def name(self):
        mask_name = 'M' if self.support_no_streaming_mask else ''
        return f"{self._name}{mask_name}"

    @staticmethod
    def create(support_no_streaming_mask: bool):
        raise NotImplementedError()

    def generate_no_stream_mask(self, generator: 'Generator'):
        if not generator.launcher_hooked('no_stream_mask'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation.streaming, 'no_stream_mask')")
            generator.launcher_hook('no_stream_mask', 'const at::Tensor no_stream_mask',
                                    'no_stream_mask', 'simulation.streaming.no_stream_mask')
        if not generator.kernel_hooked('no_stream_mask'):
            generator.kernel_hook('no_stream_mask', 'const byte_t* no_stream_mask', 'no_stream_mask.data<byte_t>()')

    def generate_read_write(self, generator: 'Generator'):
        raise NotImplementedError()


class NativeNoStreaming(NativeStreaming):
    _name = 'no'

    def __init__(self):
        super().__init__()

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeNoStreaming()

    def generate_read_write(self, generator: 'Generator'):
        if not generator.registered('read_write()'):
            generator.register('read_write()')

            # dependencies:
            generator.stencil.generate_q(generator)
            generator.cuda.generate_offset(generator)

            # read
            generator.append_index_buffer('                                           ')
            generator.append_index_buffer('    scalar_t f_reg[q];                     ')
            generator.append_index_buffer('                                           ')
            generator.append_index_buffer('#pragma unroll                             ')
            generator.append_index_buffer('    for (index_t i = 0; i < q; ++i) {      ')
            generator.append_index_buffer('                                           ')
            generator.append_write_buffer('      index_t x = offset[0] * i;           ')
            generator.append_index_buffer('                                           ')
            generator.append_index_buffer('#pragma unroll                             ')
            generator.append_index_buffer('      for (index_t j = 0; j < d; ++j) {    ')
            generator.append_index_buffer('        x += offset[j + 1] * index[j];     ')
            generator.append_index_buffer('      }                                    ')
            generator.append_index_buffer('                                           ')
            generator.append_index_buffer('      f_reg[i] = f[x];                     ')
            generator.append_index_buffer('    }                                      ')
            generator.append_index_buffer('                                           ')

            # write
            generator.append_write_buffer('                                           ')
            generator.append_write_buffer('#pragma unroll                             ')
            generator.append_write_buffer('    for (index_t i = 0; i < q; ++i) {      ')
            generator.append_write_buffer('                                           ')
            generator.append_write_buffer('      index_t x = offset[0] * i;           ')
            generator.append_write_buffer('                                           ')
            generator.append_write_buffer('#pragma unroll                             ')
            generator.append_write_buffer('      for (index_t j = 0; j < d; ++j) {    ')
            generator.append_write_buffer('        x += offset[j + 1] * index[j];     ')
            generator.append_write_buffer('      }                                    ')
            generator.append_write_buffer('                                           ')
            generator.append_write_buffer('      f_next[x] = f[i];                    ')
            generator.append_write_buffer('    }                                      ')
            generator.append_write_buffer('                                           ')


class NativeStandardStreaming(NativeStreaming):
    _name = 'standard'

    def __init__(self, support_no_streaming_mask=False):
        super().__init__(support_no_streaming_mask)

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeStandardStreaming(support_no_streaming_mask)

    def generate_f_next(self, generator: 'Generator'):
        if not generator.registered('f_next'):
            generator.register('f_next')

            generator.append_python_wrapper_after_buffer(
                'simulation.f, simulation.f_next = simulation.f_next, simulation.f')

            # generate code
            if not generator.launcher_hooked('f_next'):
                generator.append_python_wrapper_before_buffer("assert hasattr(simulation, 'f_next')")
                generator.launcher_hook('f_next', 'at::Tensor f_next', 'f_next', 'simulation.f_next')
            if not generator.kernel_hooked('f_next'):
                generator.kernel_hook('f_next', 'scalar_t *f_next', 'f_next.data<scalar_t>()')

    def generate_read_write(self, generator: 'Generator'):
        if not generator.registered('read_write()'):
            generator.register('read_write()')

            # dependencies:

            if self.support_no_streaming_mask:
                self.generate_no_stream_mask(generator)

            self.generate_f_next(generator)
            generator.cuda.generate_offset(generator)
            generator.cuda.generate_index(generator)
            generator.cuda.generate_dimension(generator)
            generator.stencil.generate_q(generator)

            # read
            generator.append_write_buffer('                                           ')
            generator.append_index_buffer('    scalar_t f_reg[q];                     ')
            generator.append_write_buffer('                                           ')
            generator.append_index_buffer('#pragma unroll                             ')
            generator.append_index_buffer('    for (index_t i = 0; i < q; ++i) {      ')
            generator.append_write_buffer('                                           ')
            generator.append_index_buffer('      index_t x = offset[0] * i;           ')
            generator.append_write_buffer('                                           ')
            generator.append_index_buffer('#pragma unroll                             ')
            generator.append_index_buffer('      for (index_t j = 0; j < d; ++j) {    ')
            generator.append_index_buffer('        x += offset[j + 1] * index[j];     ')
            generator.append_index_buffer('      }                                    ')
            generator.append_index_buffer('                                           ')
            generator.append_index_buffer('      f_reg[i] = f[x];                     ')
            generator.append_index_buffer('    }                                      ')
            generator.append_write_buffer('                                           ')

            # write and stream
            generator.append_write_buffer('                                                          ')
            generator.append_write_buffer('#pragma unroll                                            ')
            generator.append_write_buffer('    for (index_t i = 0; i < q; ++i) {                     ')
            generator.append_write_buffer('                                                          ')
            generator.append_index_buffer('      index_t x = offset[0] * i;                          ')
            generator.append_write_buffer('                                                          ')
            generator.append_write_buffer('#pragma unroll                                            ')
            generator.append_write_buffer('      for (index_t j = 0; j < d; ++j) {                   ')
            generator.append_write_buffer('                                                          ')
            generator.append_write_buffer('        // index with streaming                           ')
            generator.append_write_buffer('        index_t x_ = index[j] + e[i][j];                  ')
            generator.append_write_buffer('                                                          ')
            generator.append_write_buffer('        // correct streaming border                       ')
            generator.append_write_buffer('             if (x <  0)            x += dimension[j];    ')
            generator.append_write_buffer('        else if (x >= dimension[j]) x -= dimension[j];    ')
            generator.append_write_buffer('                                                          ')
            generator.append_write_buffer('        x += offset[j+1] * x_;                            ')
            generator.append_write_buffer('      }                                                   ')
            generator.append_write_buffer('                                                          ')
            generator.append_write_buffer('      f_next[x] = f[i];                                   ')
            generator.append_write_buffer('    }                                                     ')
            generator.append_write_buffer('                                                          ')
