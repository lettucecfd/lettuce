from . import *


class NativeStreaming(NativeLatticeBase):

    def __init__(self):
        NativeLatticeBase.__init__(self)

    # noinspection PyMethodMayBeStatic
    def generate_no_stream_mask(self, generator: 'Generator'):
        if not generator.launcher_hooked('no_stream_mask'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation.streaming, 'no_stream_mask')")
            generator.launcher_hook('no_stream_mask', 'const at::Tensor no_stream_mask', 'no_stream_mask', 'simulation.streaming.no_stream_mask')
        if not generator.kernel_hooked('no_stream_mask'):
            generator.kernel_hook('no_stream_mask', 'const byte_t* no_stream_mask', 'no_stream_mask.data<byte_t>()')

    # noinspection PyMethodMayBeStatic
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


class NativeRead(NativeStreaming):

    def __init__(self):
        NativeStreaming.__init__(self)

    @property
    def name(self) -> str:
        return f"Read"

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeRead()

    # noinspection PyMethodMayBeStatic
    def generate_f_reg(self, generator: 'Generator'):
        if not generator.registered('f_reg'):
            generator.register('f_reg')

            # dependencies:
            generator.stencil.generate_q(generator)
            generator.cuda.generate_index(generator)

            d = generator.stencil.stencil.D()
            coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])

            # read
            global_buf = generator.append_global_buffer
            global_buf(f'  scalar_t f_reg[q];              ')
            global_buf(f'#pragma unroll                    ')
            global_buf(f'  for (index_t i = 0; i < q; ++i) ')
            global_buf(f'  {{                              ')
            global_buf(f'    const index_t q_ = i;         ')
            global_buf(f'    const index_t x_ = index[0];  ', cond=d > 0)
            global_buf(f'    const index_t y_ = index[1];  ', cond=d > 1)
            global_buf(f'    const index_t z_ = index[2];  ', cond=d > 2)
            global_buf(f'    f_reg[i] = f[{coord}];        ')
            global_buf(f'  }}                              ')

    def generate_read(self, generator: 'Generator'):
        self.generate_f_reg(generator)


class NativeWrite(NativeStreaming):
    use_f_next: bool

    def __init__(self, use_f_next: bool = True):
        NativeStreaming.__init__(self)
        self.use_f_next = use_f_next

    @property
    def name(self) -> str:
        next_name = 'Next' if self.use_f_next else ''
        return f"Write{next_name}"

    @staticmethod
    def create(support_no_streaming_mask: bool, use_f_next: bool = True):
        return NativeWrite(use_f_next)

    def generate_write(self, generator: 'Generator'):
        # dependencies:
        generator.read.generate_f_reg(generator)
        generator.stencil.generate_q(generator)
        generator.cuda.generate_index(generator)
        if self.use_f_next:
            self.generate_f_next(generator)

        d = generator.stencil.stencil.D()
        coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])

        # write
        pipe_buf = generator.append_pipeline_buffer
        pipe_buf(f'#pragma unroll                        ')
        pipe_buf(f'  for (index_t i = 0; i < q; ++i) {{  ')
        pipe_buf(f'    const index_t q_ = i;             ')
        pipe_buf(f'    const index_t x_ = index[0];      ', cond=d > 0)
        pipe_buf(f'    const index_t y_ = index[1];      ', cond=d > 1)
        pipe_buf(f'    const index_t z_ = index[2];      ', cond=d > 2)
        pipe_buf(f'    f[{coord}] = f_reg[i];            ', cond=not self.use_f_next)
        pipe_buf(f'    f_next[{coord}] = f_reg[i];       ', cond=self.use_f_next)
        pipe_buf(f'  }}                                  ')


class NativeStandardStreamingWrite(NativeWrite):
    support_no_streaming_mask: bool
    use_f_next: bool

    def __init__(self, support_no_streaming_mask=False, use_f_next: bool = True):
        NativeWrite.__init__(self)
        self.support_no_streaming_mask = support_no_streaming_mask
        self.use_f_next = use_f_next

    @property
    def name(self) -> str:
        mask_name = 'Masked' if self.support_no_streaming_mask else ''
        next_name = 'Next' if self.use_f_next else ''
        return f"{mask_name}StandardStreamingWrite{next_name}"

    @staticmethod
    def create(support_no_streaming_mask: bool, use_f_next: bool = True):
        return NativeStandardStreamingWrite(support_no_streaming_mask, use_f_next)

    def generate_write(self, generator: 'Generator'):
        # dependencies:
        if self.support_no_streaming_mask:
            self.generate_no_stream_mask(generator)

        generator.stencil.generate_q(generator)
        generator.cuda.generate_index(generator)
        generator.cuda.generate_dimension(generator)
        generator.stencil.generate_e(generator)
        if self.use_f_next:
            self.generate_f_next(generator)

        d = generator.stencil.stencil.D()
        coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])
        mask_coord = generator.lattice.get_mask_coordinate(generator, ['index[0]', 'index[1]', 'index[2]'][:d])

        # write
        pipe_buf = generator.append_pipeline_buffer
        pipe_buf(f'                                                         ')
        pipe_buf(f'#pragma unroll                                           ')
        pipe_buf(f'  for (index_t i = 0; i < q; ++i) {{                     ')
        pipe_buf(f'                                                         ')
        pipe_buf(f'    const index_t q_ = i;                                ')
        pipe_buf(f'                                                         ', cond=self.support_no_streaming_mask)
        pipe_buf(f'    if (no_stream_mask[{mask_coord}]) {{                 ', cond=self.support_no_streaming_mask)
        pipe_buf(f'                                                         ', cond=self.support_no_streaming_mask)
        pipe_buf(f'      const index_t x_ = index[0];                       ', cond=self.support_no_streaming_mask and d > 0)
        pipe_buf(f'      const index_t y_ = index[1];                       ', cond=self.support_no_streaming_mask and d > 1)
        pipe_buf(f'      const index_t z_ = index[2];                       ', cond=self.support_no_streaming_mask and d > 2)
        pipe_buf(f'                                                         ', cond=self.support_no_streaming_mask)
        pipe_buf(f'      f_next[{coord}] = f_reg[i];                        ', cond=self.support_no_streaming_mask)
        pipe_buf(f'                                                         ', cond=self.support_no_streaming_mask)
        pipe_buf(f'    }} else {{                                           ', cond=self.support_no_streaming_mask)
        pipe_buf(f'                                                         ')
        pipe_buf(f'      index_t x_ = index[0] + e[i][0];                   ', cond=d > 0)
        pipe_buf(f'            if (x_ <  0)            x_ += dimension[0];  ', cond=d > 0)
        pipe_buf(f'       else if (x_ >= dimension[0]) x_ -= dimension[0];  ', cond=d > 0)
        pipe_buf(f'                                                         ', cond=d > 0)
        pipe_buf(f'      index_t y_ = index[1] + e[i][1];                   ', cond=d > 1)
        pipe_buf(f'            if (y_ <  0)            y_ += dimension[1];  ', cond=d > 1)
        pipe_buf(f'       else if (y_ >= dimension[1]) y_ -= dimension[1];  ', cond=d > 1)
        pipe_buf(f'                                                         ', cond=d > 1)
        pipe_buf(f'      index_t z_ = index[2] + e[i][2];                   ', cond=d > 2)
        pipe_buf(f'            if (z_ <  0)            z_ += dimension[2];  ', cond=d > 2)
        pipe_buf(f'       else if (z_ >= dimension[2]) z_ -= dimension[2];  ', cond=d > 2)
        pipe_buf(f'                                                         ', cond=d > 2)
        pipe_buf(f'      f[{coord}] = f_reg[i];                             ', cond=not self.use_f_next)
        pipe_buf(f'      f_next[{coord}] = f_reg[i];                        ', cond=self.use_f_next)
        pipe_buf(f'                                                         ')
        pipe_buf(f'    }}                                                   ', cond=self.support_no_streaming_mask)
        pipe_buf(f'  }}                                                     ')
        pipe_buf(f'                                                         ')


class NativeStandardStreamingRead(NativeRead):
    support_no_streaming_mask: bool

    def __init__(self, support_no_streaming_mask=False):
        NativeRead.__init__(self)
        self.support_no_streaming_mask = support_no_streaming_mask

    @property
    def name(self) -> str:
        mask_name = 'Masked' if self.support_no_streaming_mask else ''
        return f"{mask_name}StandardStreamingRead"

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeStandardStreamingRead(support_no_streaming_mask)

    def generate_f_reg(self, generator: 'Generator'):
        if not generator.registered('f_reg'):
            generator.register('f_reg')

            # dependencies:
            if self.support_no_streaming_mask:
                self.generate_no_stream_mask(generator)

            generator.stencil.generate_q(generator)
            generator.cuda.generate_index(generator)
            generator.cuda.generate_dimension(generator)
            generator.stencil.generate_e(generator)
            self.generate_f_next(generator)

            d = generator.stencil.stencil.D()
            coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])
            mask_coord = generator.lattice.get_mask_coordinate(generator, ['index[0]', 'index[1]', 'index[2]'][:d])

            # write
            global_buf = generator.append_global_buffer
            global_buf(f'  scalar_t f_reg[q];                                     ')
            global_buf(f'#pragma unroll                                           ')
            global_buf(f'  for (index_t i = 0; i < q; ++i) {{                     ')
            global_buf(f'                                                         ')
            global_buf(f'    const index_t q_ = i;                                ')
            global_buf(f'                                                         ', cond=self.support_no_streaming_mask)
            global_buf(f'    if (no_stream_mask[{mask_coord}]) {{                 ', cond=self.support_no_streaming_mask)
            global_buf(f'                                                         ', cond=self.support_no_streaming_mask)
            global_buf(f'      const index_t x_ = index[0];                       ', cond=self.support_no_streaming_mask and d > 0)
            global_buf(f'      const index_t y_ = index[1];                       ', cond=self.support_no_streaming_mask and d > 1)
            global_buf(f'      const index_t z_ = index[2];                       ', cond=self.support_no_streaming_mask and d > 2)
            global_buf(f'                                                         ', cond=self.support_no_streaming_mask)
            global_buf(f'      f_reg[i] = f[{coord}];                             ', cond=self.support_no_streaming_mask)
            global_buf(f'                                                         ', cond=self.support_no_streaming_mask)
            global_buf(f'    }} else {{                                           ', cond=self.support_no_streaming_mask)
            global_buf(f'                                                         ')
            global_buf(f'      index_t x_ = index[0] - e[i][0];                   ', cond=d > 0)
            global_buf(f'            if (x_ <  0)            x_ += dimension[0];  ', cond=d > 0)
            global_buf(f'       else if (x_ >= dimension[0]) x_ -= dimension[0];  ', cond=d > 0)
            global_buf(f'                                                         ', cond=d > 0)
            global_buf(f'      index_t y_ = index[1] - e[i][1];                   ', cond=d > 1)
            global_buf(f'            if (y_ <  0)            y_ += dimension[1];  ', cond=d > 1)
            global_buf(f'       else if (y_ >= dimension[1]) y_ -= dimension[1];  ', cond=d > 1)
            global_buf(f'                                                         ', cond=d > 1)
            global_buf(f'      index_t z_ = index[2] - e[i][2];                   ', cond=d > 2)
            global_buf(f'            if (z_ <  0)            z_ += dimension[2];  ', cond=d > 2)
            global_buf(f'       else if (z_ >= dimension[2]) z_ -= dimension[2];  ', cond=d > 2)
            global_buf(f'                                                         ', cond=d > 2)
            global_buf(f'      f_reg[i] = f[{coord}];                             ')
            global_buf(f'                                                         ')
            global_buf(f'    }}                                                   ', cond=self.support_no_streaming_mask)
            global_buf(f'  }}                                                     ')
            global_buf(f'                                                         ')
