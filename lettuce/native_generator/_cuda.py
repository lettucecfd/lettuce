from . import *


class NativeCuda:
    """
    This class provides variables that scale the Cuda kernel (thread_count and block_count).
    A Cuda kernel implicitly provides the variables blockIdx, blockDim and threadIdx.
    These variables and variables calculated from them are also provided by this class.
    All of these variables have the type index_t (aka unsigned int)
    or dim3 (struct of three index_t variables).
    """

    # noinspection PyMethodMayBeStatic
    def generate_thread_count(self, generator: 'Generator'):
        if not generator.registered('thread_count'):
            generator.register('thread_count')

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"
            assert d <= 3, "Method is undefined fot this Parameter"

            # we target 512 threads at the moment
            launcher_buf = generator.append_launcher_buffer
            launcher_buf('  const auto thread_count = dim3{16u};         ', cond=d == 1)
            launcher_buf('  const auto thread_count = dim3{16u, 16u};    ', cond=d == 2)
            launcher_buf('  const auto thread_count = dim3{8u, 8u, 8u};  ', cond=d == 3)

    def generate_block_count(self, generator: 'Generator'):
        if not generator.registered('block_count'):
            generator.register('block_count')

            # dependencies
            self.generate_thread_count(generator)
            self.generate_dimension(generator)

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"
            assert d <= 3, "Method is undefined fot this Parameter"

            # generate
            launcher_buf = generator.append_launcher_buffer

            launcher_buf('  assert((dimension[0] % thread_count.x) == 0u);  ')
            launcher_buf('  assert((dimension[1] % thread_count.y) == 0u);  ', cond=d > 1)
            launcher_buf('  assert((dimension[2] % thread_count.z) == 0u);  ', cond=d > 2)

            launcher_buf('  const auto block_count = dim3{    ')
            launcher_buf('    dimension[0] / thread_count.x,  ')
            launcher_buf('    dimension[1] / thread_count.y,  ', cond=d > 1)
            launcher_buf('    dimension[2] / thread_count.z,  ', cond=d > 2)
            launcher_buf('  };                                ')

    # noinspection PyMethodMayBeStatic
    def generate_index(self, generator: 'Generator'):
        if not generator.registered('index'):
            generator.register('index')

            # dependencies
            generator.stencil.generate_d(generator)

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"
            assert d <= 3, "Method is undefined fot this Parameter"

            # generate
            index_buf = generator.append_index_buffer
            index_buf('  const index_t index[d] = {                                      ')
            index_buf('    static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x),  ')
            index_buf('    static_cast<index_t>(blockIdx.y * blockDim.y + threadIdx.y),  ', cond=d > 1)
            index_buf('    static_cast<index_t>(blockIdx.z * blockDim.z + threadIdx.z),  ', cond=d > 2)
            index_buf('  };                                                              ')

    # noinspection PyMethodMayBeStatic
    def generate_dimension(self, generator: 'Generator'):
        if not generator.registered('dimension'):
            generator.register('dimension')

            # dependencies
            generator.stencil.generate_d(generator)

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"
            assert d <= 3, "Method is undefined fot this Parameter"

            # generate launcher
            launcher_buf = generator.append_launcher_buffer
            launcher_buf('  const index_t dimension[d] = {          ')
            launcher_buf('    static_cast<index_t> (f.sizes()[1]),  ')
            launcher_buf('    static_cast<index_t> (f.sizes()[2]),  ', cond=d > 1)
            launcher_buf('    static_cast<index_t> (f.sizes()[3]),  ', cond=d > 2)
            launcher_buf('  };                                      ')

            # hook into kernel
            kernel_hook = generator.kernel_hook
            kernel_hook('dimension0', 'index_t dimension0', 'dimension[0]')
            kernel_hook('dimension1', 'index_t dimension1', 'dimension[1]', cond=d > 1)
            kernel_hook('dimension2', 'index_t dimension2', 'dimension[2]', cond=d > 2)

            # generate kernel
            index_buf = generator.append_index_buffer
            index_buf('  const index_t dimension[d] = {  ')
            index_buf('    dimension0,                   ')
            index_buf('    dimension1,                   ', cond=d > 1)
            index_buf('    dimension2,                   ', cond=d > 2)
            index_buf('  };                              ')
