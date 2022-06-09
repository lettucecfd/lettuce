from . import *
from abc import abstractmethod


class NativeStreaming(NativeLatticeBase):
    support_no_streaming_mask: bool

    def __init__(self, support_no_streaming_mask=False):
        self.support_no_streaming_mask = support_no_streaming_mask

    @staticmethod
    @abstractmethod
    def create(support_no_streaming_mask: bool):
        ...

    # noinspection PyMethodMayBeStatic
    def generate_no_stream_mask(self, generator: 'Generator'):
        if not generator.launcher_hooked('no_stream_mask'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation.streaming, 'no_stream_mask')")
            generator.launcher_hook('no_stream_mask', 'const at::Tensor no_stream_mask', 'no_stream_mask', 'simulation.streaming.no_stream_mask')
        if not generator.kernel_hooked('no_stream_mask'):
            generator.kernel_hook('no_stream_mask', 'const byte_t* no_stream_mask', 'no_stream_mask.data<byte_t>()')

    @abstractmethod
    def generate_read(self, generator: 'Generator'):
        ...

    @abstractmethod
    def generate_write(self, generator: 'Generator'):
        ...

    def generate_read_write(self, generator: 'Generator'):
        self.generate_read(generator)
        self.generate_write(generator)


class NativeNoStreaming(NativeStreaming):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return 'NoStreaming'

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeNoStreaming()

    def generate_read(self, generator: 'Generator'):
        if not generator.registered('read()'):
            generator.register('read()')

            # dependencies:
            generator.stencil.generate_q(generator)
            generator.cuda.generate_index(generator)

            d = generator.stencil.stencil.D()
            coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])

            # read
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'  scalar_t f_reg[q];                  ')
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'#pragma unroll                        ')
            generator.append_index_buffer(f'  for (index_t i = 0; i < q; ++i) {{  ')
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'    const index_t q_ = i;             ')
            generator.append_index_buffer(f'    const index_t x_ = index[0];      ', cond=d > 0)
            generator.append_index_buffer(f'    const index_t y_ = index[1];      ', cond=d > 1)
            generator.append_index_buffer(f'    const index_t z_ = index[2];      ', cond=d > 2)
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'    f_reg[i] = f[{coord}];            ')
            generator.append_index_buffer(f'  }}                                  ')
            generator.append_index_buffer(f'                                      ')

    def generate_write(self, generator: 'Generator'):
        if not generator.registered('write()'):
            generator.register('write()')

            # dependencies:
            generator.stencil.generate_q(generator)
            generator.cuda.generate_index(generator)

            d = generator.stencil.stencil.D()
            coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])

            # write
            generator.append_write_buffer(f'                                      ')
            generator.append_write_buffer(f'#pragma unroll                        ')
            generator.append_write_buffer(f'  for (index_t i = 0; i < q; ++i) {{  ')
            generator.append_write_buffer(f'                                      ')
            generator.append_write_buffer(f'    const index_t q_ = i;             ')
            generator.append_write_buffer(f'    const index_t x_ = index[0];      ', cond=d > 0)
            generator.append_write_buffer(f'    const index_t y_ = index[1];      ', cond=d > 1)
            generator.append_write_buffer(f'    const index_t z_ = index[2];      ', cond=d > 2)
            generator.append_write_buffer(f'                                      ')
            generator.append_write_buffer(f'    f[{coord}] = f_reg[i];            ')
            generator.append_write_buffer(f'  }}                                  ')
            generator.append_write_buffer(f'                                      ')


class NativeStandardStreaming(NativeStreaming):
    _name = 'standard'

    def __init__(self, support_no_streaming_mask=False):
        super().__init__(support_no_streaming_mask)

    @property
    def name(self) -> str:
        mask_name = 'Masked' if self.support_no_streaming_mask else ''
        return f"{mask_name}StandardStreaming"

    @staticmethod
    def create(support_no_streaming_mask: bool):
        return NativeStandardStreaming(support_no_streaming_mask)

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

    def generate_read(self, generator: 'Generator'):
        if not generator.registered('read()'):
            generator.register('read()')

            # dependencies:
            generator.cuda.generate_index(generator)
            generator.stencil.generate_q(generator)

            d = generator.stencil.stencil.D()
            coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])

            # read
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'  scalar_t f_reg[q];                  ')
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'#pragma unroll                        ')
            generator.append_index_buffer(f'  for (index_t i = 0; i < q; ++i) {{  ')
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'    const index_t q_ = i;             ')
            generator.append_index_buffer(f'    const index_t x_ = index[0];      ', cond=d > 0)
            generator.append_index_buffer(f'    const index_t y_ = index[1];      ', cond=d > 1)
            generator.append_index_buffer(f'    const index_t z_ = index[2];      ', cond=d > 2)
            generator.append_index_buffer(f'                                      ')
            generator.append_index_buffer(f'    f_reg[i] = f[{coord}];            ')
            generator.append_index_buffer(f'  }}                                  ')
            generator.append_index_buffer(f'                                      ')

    def generate_write(self, generator: 'Generator'):
        if not generator.registered('write()'):
            generator.register('write()')

            # dependencies:
            if self.support_no_streaming_mask:
                self.generate_no_stream_mask(generator)

            generator.stencil.generate_q(generator)
            generator.cuda.generate_index(generator)
            generator.stencil.generate_e(generator)
            generator.cuda.generate_dimension(generator)
            self.generate_f_next(generator)

            d = generator.stencil.stencil.D()
            coord = generator.lattice.get_lattice_coordinate(generator, ['q_', 'x_', 'y_', 'z_'][:d + 1])
            mask_coord = generator.lattice.get_mask_coordinate(generator, ['index[0]', 'index[1]', 'index[2]'][:d])

            # write
            generator.append_write_buffer(f'                                                         ')
            generator.append_write_buffer(f'#pragma unroll                                           ')
            generator.append_write_buffer(f'  for (index_t i = 0; i < q; ++i) {{                     ')
            generator.append_write_buffer(f'                                                         ')
            generator.append_write_buffer(f'    const index_t q_ = i;                                ')
            generator.append_write_buffer(f'                                                         ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'    if (no_stream_mask[{mask_coord}]) {{                 ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'                                                         ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'      const index_t x_ = index[0];                       ', cond=self.support_no_streaming_mask and d > 0)
            generator.append_write_buffer(f'      const index_t y_ = index[1];                       ', cond=self.support_no_streaming_mask and d > 1)
            generator.append_write_buffer(f'      const index_t z_ = index[2];                       ', cond=self.support_no_streaming_mask and d > 2)
            generator.append_write_buffer(f'                                                         ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'      f_next[{coord}] = f_reg[i];                        ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'                                                         ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'    }} else {{                                           ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'                                                         ')
            generator.append_write_buffer(f'      index_t x_ = index[0] + e[i][0];                   ', cond=d > 0)
            generator.append_write_buffer(f'            if (x_ <  0)            x_ += dimension[0];  ', cond=d > 0)
            generator.append_write_buffer(f'       else if (x_ >= dimension[0]) x_ -= dimension[0];  ', cond=d > 0)
            generator.append_write_buffer(f'                                                         ', cond=d > 0)
            generator.append_write_buffer(f'      index_t y_ = index[1] + e[i][1];                   ', cond=d > 1)
            generator.append_write_buffer(f'            if (y_ <  0)            y_ += dimension[1];  ', cond=d > 1)
            generator.append_write_buffer(f'       else if (y_ >= dimension[1]) y_ -= dimension[1];  ', cond=d > 1)
            generator.append_write_buffer(f'                                                         ', cond=d > 1)
            generator.append_write_buffer(f'      index_t z_ = index[2] + e[i][2];                   ', cond=d > 2)
            generator.append_write_buffer(f'            if (z_ <  0)            z_ += dimension[2];  ', cond=d > 2)
            generator.append_write_buffer(f'       else if (z_ >= dimension[2]) z_ -= dimension[2];  ', cond=d > 2)
            generator.append_write_buffer(f'                                                         ', cond=d > 2)
            generator.append_write_buffer(f'      f_next[{coord}] = f_reg[i];                        ')
            generator.append_write_buffer(f'                                                         ')
            generator.append_write_buffer(f'    }}                                                   ', cond=self.support_no_streaming_mask)
            generator.append_write_buffer(f'  }}                                                     ')
            generator.append_write_buffer(f'                                                         ')
