from lettuce.gen_native import *

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


class NativeStreaming:
    """
    """

    name = 'invalidStreaming'
    read_frame_: str = read_frame
    write_frame_: str = write_frame

    @staticmethod
    def __init__():
        raise NotImplementedError("This class is not meant to be constructed "
                                  "as it provides only static fields and methods!")

    @staticmethod
    def no_stream_mask(gen: 'GeneratorKernel'):
        """
        """
        if not gen.wrapper_hooked('no_stream_mask'):
            gen.pyr("assert hasattr(simulation.streaming, 'no_stream_mask')")
            gen.wrapper_hook('no_stream_mask', 'const at::Tensor no_stream_mask',
                             'no_stream_mask', 'simulation.streaming.no_stream_mask')
        if not gen.kernel_hooked('no_stream_mask'):
            gen.kernel_hook('no_stream_mask', 'const byte_t* no_stream_mask', 'no_stream_mask.data<byte_t>()')

    @staticmethod
    def no_collision_mask(gen: 'GeneratorKernel'):
        """
        """
        if not gen.wrapper_hooked('no_collision_mask'):
            gen.pyr("assert hasattr(simulation, 'no_collision_mask')")
            gen.wrapper_hook('no_collision_mask', 'const at::Tensor no_collision_mask',
                             'no_collision_mask', 'simulation.no_collision_mask')
        if not gen.kernel_hooked('no_collision_mask'):
            gen.kernel_hook('no_collision_mask', 'const byte_t* no_collision_mask', 'no_collision_mask.data<byte_t>()')

    @staticmethod
    def read_write(gen: 'GeneratorKernel', support_no_stream: bool, support_no_collision: bool):
        raise NotImplementedError("This method is only implemented by concrete subclasses!")
