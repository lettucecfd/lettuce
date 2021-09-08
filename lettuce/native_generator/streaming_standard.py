from . import *


class NativeStreamingStandard(NativeStreaming):
    """
    """

    _name = 'standardStreaming'

    def f_next(self, generator: 'GeneratorKernel'):
        """
        """
        if not generator.registered('f_next'):
            generator.register('f_next')

            generator.pyo('simulation.f, simulation.f_next = simulation.f_next, simulation.f')

            # generate code
            if not generator.wrapper_hooked('f_next'):
                generator.pyr("assert hasattr(simulation, 'f_next')")
                generator.wrapper_hook('f_next', 'at::Tensor f_next', 'f_next', 'simulation.f_next')
            if not generator.kernel_hooked('f_next'):
                generator.kernel_hook('f_next', 'scalar_t *f_next', 'f_next.data<scalar_t>()')

    def dim_offset(self, generator: 'GeneratorKernel', d: int):
        """
        """
        if not generator.registered(f"dim{d}_offset"):
            generator.register(f"dim{d}_offset")

            # dependencies
            generator.cuda.index(generator, d)
            generator.cuda.dimension(generator, d, hook_into_kernel=True)

            # generate

            if d > 0:
                generator.cuda.length(generator, d - 1, hook_into_kernel=True)

                generator.idx(f"const index_t &dim{d}_offset0 = index{d} * length{d - 1};")
                generator.idx(f"const index_t dim{d}_offset1 = (((index{d} + 1) == dimension{d}) "
                        f"? 0 : (index{d} + 1)) * length{d - 1};")
                generator.idx(f"const index_t dim{d}_offset2 = ((index{d} == 0) "
                        f"? dimension{d} - 1 : (index{d} - 1)) * length{d - 1};")

            else:
                generator.idx(f"const index_t &dim0_offset0 = index0;")
                generator.idx(f"const index_t dim0_offset1 = (((index0 + 1) == dimension0) ? 0 : (index0 + 1));")
                generator.idx(f"const index_t dim0_offset2 = ((index0 == 0) ? dimension0 - 1 : (index0 - 1));")

    def read_write(self, generator: 'GeneratorKernel', support_no_stream: bool, support_no_collision: bool):
        """
        """
        if not generator.registered('read_write()'):
            generator.register('read_write()')

            # dependencies:

            if support_no_stream:
                self.no_stream_mask(generator)

            if support_no_collision:
                self.no_collision_mask(generator)

            self.f_next(generator)
            generator.stencil.q(generator)
            generator.cuda.offset(generator)
            generator.cuda.length(generator, d=generator.stencil.d_ - 1, hook_into_kernel=True)

            for d in range(generator.stencil.d_):
                self.dim_offset(generator, d)

            # read with stream
            length_index = generator.stencil.d_ - 1

            direction_slot = {0: 0, -1: 1, 1: 2}
            offsets = []
            for q in range(generator.stencil.q_):
                offset = []
                all_zero = True
                for d in range(generator.stencil.d_):
                    offset.append(f"dim{d}_offset{direction_slot[generator.stencil.e_[q][generator.stencil.d_ - 1 - d]]}")
                    all_zero = all_zero and (generator.stencil.e_[q][generator.stencil.d_ - 1 - d] == 0)

                if all_zero:
                    offsets.append('offset')
                else:
                    offsets.append(' + '.join(offset))

            generator.idx()
            generator.idx('scalar_t f_reg[q];')
            generator.idx('{')

            # stream index 0

            if support_no_stream:
                if offsets[0] == 'offset':
                    generator.idx(f'    f_reg[0] = f[offset];')
                else:
                    generator.idx(f'    f_reg[0] = no_stream_mask[offset] ? f[offset] : f[{offsets[0]}];')
            else:
                generator.idx(f"    f_reg[0] = f[{offsets[0]}];")

            # stream index 1

            index_it = f"auto index_it = length{length_index};"

            if support_no_stream:
                if offsets[1] == 'offset':
                    generator.idx(f"    {index_it} f_reg[1] = f[index_it + offset];")
                else:
                    generator.idx(f"    {index_it} {{ const auto abs_index = index_it + offset; f_reg[1] = "
                            f"no_stream_mask[abs_index] ? f[abs_index] : f[index_it + {offsets[1]}]; }}")
            else:
                generator.idx(f"    {index_it} f_reg[1] = f[index_it + {offsets[1]}];")

            # stream index n

            index_it = f"index_it += length{length_index};"

            for q in range(2, generator.stencil.q_):

                if support_no_stream:
                    if offsets[q] == 'offset':
                        generator.idx(f"    {index_it}     f_reg[{q}] = f[index_it + offset];")
                    else:
                        generator.idx(f"    {index_it}     {{ const auto abs_index = index_it + offset; f_reg[{q}] = "
                                f"no_stream_mask[abs_index] ? f[abs_index] : f[index_it + {offsets[q]}]; }}")
                else:
                    generator.idx(f"    {index_it}     f_reg[{q}] = f[index_it + {offsets[q]}];")

            generator.idx('}')
            generator.idx()

            if support_no_collision:
                generator.idx(f"if(!no_collision_mask[offset])")
            generator.idx('{')

            # write
            generator.wrt('}')
            generator.wrt()
            generator.wrt(self.write_frame_.format(target='f_next', length_index=length_index))
