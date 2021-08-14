from lettuce.gencuda import *

read_frame = '''{{
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

write_frame = '''{{
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


class Stream:
    """
    """

    def __init__(self):
        """
        """
        self.name = 'invalid'

    def no_stream_mask(self, gen: 'KernelGenerator'):
        """
        """
        if not gen.wrapper_hooked('no_stream_mask'):
            gen.pyr("assert hasattr(simulation.streaming, 'no_stream_mask')")
            gen.wrapper_hook('no_stream_mask', 'const at::Tensor no_stream_mask',
                             'no_stream_mask', 'simulation.streaming.no_stream_mask')
        if not gen.kernel_hooked('no_stream_mask'):
            gen.kernel_hook('no_stream_mask', 'const byte_t* no_stream_mask', 'no_stream_mask.data<byte_t>()')

    def no_collision_mask(self, gen: 'KernelGenerator'):
        """
        """
        if not gen.wrapper_hooked('no_collision_mask'):
            gen.pyr("assert hasattr(simulation, 'no_collision_mask')")
            gen.wrapper_hook('no_collision_mask', 'const at::Tensor no_collision_mask',
                             'no_collision_mask', 'simulation.no_collision_mask')
        if not gen.kernel_hooked('no_collision_mask'):
            gen.kernel_hook('no_collision_mask', 'const byte_t* no_collision_mask', 'no_collision_mask.data<byte_t>()')

    def read_write(self, gen: 'KernelGenerator', support_no_stream: bool, support_no_collision: bool):
        """
        """
        assert False, "Not implemented Error"


class ReadWrite(Stream):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.name = 'rw'

    def read_write(self, gen: 'KernelGenerator', support_no_stream: bool, support_no_collision: bool):
        """
        """
        if not gen.registered('read_write()'):
            gen.register('read_write()')

            # dependencies:

            if support_no_collision:
                self.no_collision_mask(gen)

            gen.stencil.q(gen)
            gen.cuda.length(gen, gen.stencil.d_ - 1, n=True)
            gen.cuda.offset(gen)

            # read
            length_index = gen.stencil.d_ - 1

            gen.idx()
            gen.idx('scalar_t f_reg[q];')
            gen.idx(read_frame.format(source='f', length_index=length_index))

            if support_no_collision:
                indices = ', '.join([f"index{d}" for d in range(gen.stencil.d_)])
                gen.idx(f"if(!no_collision_mask[{indices}])")
            gen.idx('{')

            gen.wrt('}')
            gen.wrt()
            gen.wrt(write_frame.format(target='f', length_index=length_index))


class StandardStream(Stream):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.name = 'standard'

    def f_next(self, gen: 'KernelGenerator'):
        """
        """
        if not gen.registered('f_next'):
            gen.register('f_next')

            gen.pyo('simulation.f, simulation.f_next = simulation.f_next, simulation.f')

            # generate code
            if not gen.wrapper_hooked('f_next'):
                gen.pyr("assert hasattr(simulation, 'f_next')")
                gen.wrapper_hook('f_next', 'at::Tensor f_next', 'f_next', 'simulation.f_next')
            if not gen.kernel_hooked('f_next'):
                gen.kernel_hook('f_next', 'scalar_t *f_next', 'f_next.data<scalar_t>()')

    def dim_offset(self, gen: 'KernelGenerator', d: int):
        """
        """
        if not gen.registered(f"dim{d}_offset"):
            gen.register(f"dim{d}_offset")

            # dependencies
            gen.cuda.index(gen, d)
            gen.cuda.dimension(gen, d, n=True)

            # generate

            if d > 0:
                gen.cuda.length(gen, d - 1, n=True)

                gen.idx(f"const index_t &dim{d}_offset0 = index{d};")
                gen.idx(f"const index_t dim{d}_offset1 = (((index{d} + 1) == dimension{d}) "
                        f"? 0 : (index{d} + 1)) * length{d - 1};")
                gen.idx(f"const index_t dim{d}_offset2 = ((index{d} == 0) "
                        f"? dimension{d} - 1 : (index{d} - 1)) * length{d - 1};")

            else:
                gen.idx(f"const index_t &dim0_offset0 = index0;")
                gen.idx(f"const index_t dim0_offset1 = (((index0 + 1) == dimension0) ? 0 : (index0 + 1));")
                gen.idx(f"const index_t dim0_offset2 = ((index0 == 0) ? dimension0 - 1 : (index0 - 1));")

    def read_write(self, gen: 'KernelGenerator', support_no_stream: bool, support_no_collision: bool):
        """
        """
        if not gen.registered('read_write()'):
            gen.register('read_write()')

            # dependencies:

            if support_no_stream:
                self.no_stream_mask(gen)

            if support_no_collision:
                self.no_collision_mask(gen)

            self.f_next(gen)
            gen.stencil.q(gen)
            gen.cuda.offset(gen)
            gen.cuda.length(gen, d=gen.stencil.d_ - 1, n=True)

            for d in range(gen.stencil.d_):
                self.dim_offset(gen, d)

            # read with stream
            length_index = gen.stencil.d_ - 1

            direction_slot = {0: 0, 1: 1, -1: 2}
            offsets = []
            for q in range(gen.stencil.q_):
                offset = []
                all_zero = True
                for d in range(gen.stencil.d_):
                    offset.append(f"dim{d}_offset{direction_slot[gen.stencil.e_[q][d]]}")
                    all_zero = all_zero and (gen.stencil.e_[q][d] == 0)

                if all_zero:
                    offsets.append('offset')
                else:
                    offsets.append(' + '.join(offset))

            gen.idx()
            gen.idx('scalar_t f_reg[q];')
            gen.idx('{')

            # stream index 0

            if support_no_stream:
                if offsets[0] == 'offset':
                    gen.idx(f'    f_reg[0] = f[offset];')
                else:
                    gen.idx(f'    f_reg[0] = no_stream_mask[offset] ? f[offset] : f[{offsets[0]}];')
            else:
                gen.idx(f"    f_reg[0] = f[{offsets[0]}];")

            # stream index 1

            index_it = f"auto index_it = length{length_index};"

            if support_no_stream:
                if offsets[1] == 'offset':
                    gen.idx(f"    {index_it} f_reg[1] = f[index_it + offset];")
                else:
                    gen.idx(f"    {index_it} {{ const auto abs_index = index_it + offset; f_reg[1] = "
                            f"no_stream_mask[abs_index] ? f[abs_index] : f[index_it + {offsets[1]}]; }}")
            else:
                gen.idx(f"    {index_it} f_reg[1] = f[index_it + {offsets[1]}];")

            # stream index n

            index_it = f"index_it += length{length_index};"

            for q in range(2, gen.stencil.q_):

                if support_no_stream:
                    if offsets[q] == 'offset':
                        gen.idx(f"    {index_it}     f_reg[{q}] = f[index_it + offset];")
                    else:
                        gen.idx(f"    {index_it}     {{ const auto abs_index = index_it + offset; f_reg[{q}] = "
                                f"no_stream_mask[abs_index] ? f[abs_index] : f[index_it + {offsets[q]}]; }}")
                else:
                    gen.idx(f"    {index_it}     f_reg[{q}] = f[index_it + {offsets[q]}];")

            gen.idx('}')
            gen.idx()

            if support_no_collision:
                gen.idx(f"if(!no_collision_mask[offset])")
            gen.idx('{')

            # write
            gen.wrt('}')
            gen.wrt()
            gen.wrt(write_frame.format(target='f_next', length_index=length_index))
