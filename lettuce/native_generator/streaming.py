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

    read_frame_: str = read_frame
    write_frame_: str = write_frame

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

    def no_stream_mask(self, generator: 'GeneratorKernel'):
        if not generator.wrapper_hooked('no_stream_mask'):
            generator.pyr("assert hasattr(simulation.streaming, 'no_stream_mask')")
            generator.wrapper_hook('no_stream_mask', 'const at::Tensor no_stream_mask',
                                   'no_stream_mask', 'simulation.streaming.no_stream_mask')
        if not generator.kernel_hooked('no_stream_mask'):
            generator.kernel_hook('no_stream_mask', 'const byte_t* no_stream_mask', 'no_stream_mask.data<byte_t>()')

    def read_write(self, generator: 'GeneratorKernel'):
        raise AbstractMethodInvokedError()
