from . import *

read_frame = '''
    {{
        auto index_it = offset;
        f_reg[0] = {source}[index_it];

#pragma unroll
        for(index_t i = 1; i < q; ++i)
        {{
            index_it += length{length_index};
            f_reg[i] = {source}[index_it];
        }}
    }}
'''

write_frame = '''
    {{
        auto index_it = offset;
        {target}[index_it] = f_reg[0];

#pragma unroll
        for(index_t i = 1; i < q; ++i)
        {{
            index_it += length{length_index};
            {target}[index_it] = f_reg[i];
        }}
    }}
'''


class NativeStreaming(NativeLatticeBase):
    _name = 'invalidStreaming'

    support_no_streaming_mask: bool

    def __init__(self, support_no_streaming_mask=False):
        self.support_no_streaming_mask = support_no_streaming_mask

    @property
    def name(self):
        mask_name = 'Masked' if self.support_no_streaming_mask else ''
        return f"{self._name}{mask_name}"

    @staticmethod
    def create(support_no_streaming_mask: bool):
        raise AbstractMethodInvokedError()

    def generate_no_stream_mask(self, generator: 'GeneratorKernel'):
        if not generator.wrapper_hooked('no_stream_mask'):
            generator.pyr("assert hasattr(simulation.streaming, 'no_stream_mask')")
            generator.wrapper_hook('no_stream_mask', 'const at::Tensor no_stream_mask',
                                   'no_stream_mask', 'simulation.streaming.no_stream_mask')
        if not generator.kernel_hooked('no_stream_mask'):
            generator.kernel_hook('no_stream_mask', 'const byte_t* no_stream_mask', 'no_stream_mask.data<byte_t>()')

    def generate_read_write(self, generator: 'GeneratorKernel'):
        raise AbstractMethodInvokedError()


class NativeNoStreaming(NativeStreaming):
    _name = 'noStreaming'

    def __init__(self):
        super().__init__(False)

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeNoStreaming()

    def generate_read_write(self, generator: 'GeneratorKernel'):
        if not generator.registered('read_write()'):
            generator.register('read_write()')

            # dependencies:

            generator.stencil.generate_q(generator)
            generator.cuda.generate_length(generator, generator.stencil.stencil.d() - 1, hook_into_kernel=True)
            generator.cuda.generate_offset(generator)

            # read
            length_index = generator.stencil.stencil.d() - 1

            generator.idx()
            generator.idx('scalar_t f_reg[q];')
            generator.idx(read_frame.format(source='f', length_index=length_index))

            generator.wrt()
            generator.wrt(write_frame.format(target='f', length_index=length_index))


class NativeStandardStreaming(NativeStreaming):
    _name = 'standardStreaming'

    def __init__(self, support_no_streaming_mask=False):
        super().__init__(support_no_streaming_mask)

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeStandardStreaming(support_no_streaming_mask)

    def generate_f_next(self, generator: 'GeneratorKernel'):
        if not generator.registered('f_next'):
            generator.register('f_next')

            generator.pyo('simulation.f, simulation.f_next = simulation.f_next, simulation.f')

            # generate code
            if not generator.wrapper_hooked('f_next'):
                generator.pyr("assert hasattr(simulation, 'f_next')")
                generator.wrapper_hook('f_next', 'at::Tensor f_next', 'f_next', 'simulation.f_next')
            if not generator.kernel_hooked('f_next'):
                generator.kernel_hook('f_next', 'scalar_t *f_next', 'f_next.data<scalar_t>()')

    def generate_dim_offset(self, generator: 'GeneratorKernel', d: int):
        if not generator.registered(f"dim{d}_offset"):
            generator.register(f"dim{d}_offset")

            # dependencies
            generator.cuda.generate_index(generator, d)
            generator.cuda.generate_dimension(generator, d, hook_into_kernel=True)

            # generate

            if d > 0:
                generator.cuda.generate_length(generator, d - 1, hook_into_kernel=True)

                generator.idx(f"const index_t &dim{d}_offset0 = index{d} * length{d - 1};")
                generator.idx(f"const index_t dim{d}_offset1 = (((index{d} + 1) == dimension{d}) "
                              f"? 0 : (index{d} + 1)) * length{d - 1};")
                generator.idx(f"const index_t dim{d}_offset2 = ((index{d} == 0) "
                              f"? dimension{d} - 1 : (index{d} - 1)) * length{d - 1};")

            else:
                generator.idx(f"const index_t &dim0_offset0 = index0;")
                generator.idx(f"const index_t dim0_offset1 = (((index0 + 1) == dimension0) ? 0 : (index0 + 1));")
                generator.idx(f"const index_t dim0_offset2 = ((index0 == 0) ? dimension0 - 1 : (index0 - 1));")

    def generate_read_write(self, generator: 'GeneratorKernel'):
        if not generator.registered('read_write()'):
            generator.register('read_write()')

            # dependencies:

            if self.support_no_streaming_mask:
                self.generate_no_stream_mask(generator)

            self.generate_f_next(generator)
            generator.stencil.generate_q(generator)
            generator.cuda.generate_offset(generator)
            generator.cuda.generate_length(generator, d=generator.stencil.stencil.d() - 1, hook_into_kernel=True)

            for d in range(generator.stencil.stencil.d()):
                self.generate_dim_offset(generator, d)

            # read with stream
            length_index = generator.stencil.stencil.d() - 1

            direction_slot = {0: 0, -1: 1, 1: 2}
            offsets = []
            for q in range(generator.stencil.stencil.q()):
                offset = []
                all_zero = True
                for d in range(generator.stencil.stencil.d()):
                    offset.append(
                        f"dim{d}_offset{direction_slot[generator.stencil.stencil.e[q][generator.stencil.stencil.d() - 1 - d]]}")
                    all_zero = all_zero and (generator.stencil.stencil.e[q][generator.stencil.stencil.d() - 1 - d] == 0)

                if all_zero:
                    offsets.append('offset')
                else:
                    offsets.append(' + '.join(offset))

            generator.idx()
            generator.idx('scalar_t f_reg[q];')
            generator.idx('{')

            # stream index 0

            if self.support_no_streaming_mask:
                if offsets[0] == 'offset':
                    generator.idx(f'    f_reg[0] = f[offset];')
                else:
                    generator.idx(f'    f_reg[0] = no_stream_mask[offset] ? f[offset] : f[{offsets[0]}];')
            else:
                generator.idx(f"    f_reg[0] = f[{offsets[0]}];")

            # stream index 1

            index_it = f"auto index_it = length{length_index};"

            if self.support_no_streaming_mask:
                if offsets[1] == 'offset':
                    generator.idx(f"    {index_it} f_reg[1] = f[index_it + offset];")
                else:
                    generator.idx(f"    {index_it} {{ const auto abs_index = index_it + offset; f_reg[1] = "
                                  f"no_stream_mask[abs_index] ? f[abs_index] : f[index_it + {offsets[1]}]; }}")
            else:
                generator.idx(f"    {index_it} f_reg[1] = f[index_it + {offsets[1]}];")

            # stream index n

            index_it = f"index_it += length{length_index};"

            for q in range(2, generator.stencil.stencil.q()):

                if self.support_no_streaming_mask:
                    if offsets[q] == 'offset':
                        generator.idx(f"    {index_it}     f_reg[{q}] = f[index_it + offset];")
                    else:
                        generator.idx(f"    {index_it}     {{ const auto abs_index = index_it + offset; f_reg[{q}] = "
                                      f"no_stream_mask[abs_index] ? f[abs_index] : f[index_it + {offsets[q]}]; }}")
                else:
                    generator.idx(f"    {index_it}     f_reg[{q}] = f[index_it + {offsets[q]}];")

            generator.idx('}')
            generator.idx()

            # write
            generator.wrt()
            generator.wrt(write_frame.format(target='f_next', length_index=length_index))
