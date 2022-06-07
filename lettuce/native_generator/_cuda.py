from . import *


class NativeCuda:
    """
    This class provides variables that scale the Cuda kernel (thread_count and block_count).
    A Cuda kernel implicitly provides the variables blockIdx, blockDim and threadIdx.
    This variables and variables calculated from them are also provided by this class.
    All of this variables have the type index_t (aka unsigned int)
    or dim3 (struct of three index_t variables).
    """

    def generate_thread_count(self, generator: 'Generator'):
        if not generator.registered('thread_count'):
            generator.register('thread_count')

            # TODO find an algorithm for this instead of hard coding
            # for d in range(self.stencil.d_):
            #    # dependencies
            #    self.dimension(gen, d)

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"
            assert d <= 3, "Method is undefined fot this Parameter"

            # we target 512 threads at the moment
            if d == 1:
                generator.append_launcher_buffer("const auto thread_count = dim3{16u};")
            if d == 2:
                generator.append_launcher_buffer("const auto thread_count = dim3{16u, 16u};")
            if d == 3:
                generator.append_launcher_buffer("const auto thread_count = dim3{8u, 8u, 8u};")

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
            if d > 0:
                generator.append_launcher_buffer(f"assert((dimension[0] % thread_count.x) == 0u);")
            if d > 1:
                generator.append_launcher_buffer(f"assert((dimension[1] % thread_count.y) == 0u);")
            if d > 2:
                generator.append_launcher_buffer(f"assert((dimension[2] % thread_count.z) == 0u);")
            generator.append_launcher_buffer('const auto block_count = dim3{')
            if d > 0:
                generator.append_launcher_buffer('dimension[0] / thread_count.x,')
            if d > 1:
                generator.append_launcher_buffer('dimension[1] / thread_count.y,')
            if d > 2:
                generator.append_launcher_buffer('dimension[2] / thread_count.z,')
            generator.append_launcher_buffer('};')

    def generate_index(self, generator: 'Generator'):
        if not generator.registered('index'):
            generator.register('index')

            # dependencies
            generator.stencil.generate_d(generator)

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"
            assert d <= 3, "Method is undefined fot this Parameter"

            # generate
            generator.append_index_buffer('const index_t index[d] = {')
            if d > 0:
                generator.append_index_buffer('blockIdx.x * blockDim.x + threadIdx.x,')
            if d > 1:
                generator.append_index_buffer('blockIdx.y * blockDim.y + threadIdx.y,')
            if d > 2:
                generator.append_index_buffer('blockIdx.z * blockDim.z + threadIdx.z,')
            generator.append_index_buffer('};')

    def generate_dimension(self, generator: 'Generator'):
        if not generator.registered('dimension'):
            generator.register('dimension')

            # dependencies
            generator.stencil.generate_d(generator)

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"

            # generate
            generator.append_launcher_buffer('const index_t dimension[d] = {')
            [generator.append_launcher_buffer(f"static_cast<index_t> (f.sizes()[{i}]),") for i in range(d)]
            generator.append_launcher_buffer('}')

            # hook into kernel by default
            generator.kernel_hook('dimension', f"index_t dimension[{d}]", 'dimension')

    def generate_offset(self, generator: 'Generator'):
        """
        offset describes the offset which needs to be multiplied with a coordinate
        to get the total index in the data
        """
        if not generator.registered('offset'):
            generator.register('offset')

            # dependencies
            self.generate_dimension(generator)
            generator.stencil.generate_d(generator)

            d = generator.stencil.stencil.D()
            assert d > 0, "Method is undefined fot this Parameter"

            # generate
            generator.append_index_buffer('index_t offset[d + 1] = {')
            if d > 0:
                generator.append_index_buffer('0,')
            if d > 1:
                generator.append_index_buffer('dimension[0],')
            [generator.append_index_buffer(f"offset[{i - 1}] * dimension[{i - 1}],") for i in range(2, d)]
            generator.append_index_buffer('}')
