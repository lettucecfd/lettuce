from . import *


class NativeCuda:
    """
    This class provides variables that scale the Cuda kernel (thread_count and block_count).
    A Cuda kernel implicitly provides the variables blockIdx, blockDim and threadIdx.
    This variables and variables calculated from them are also provided by this class.
    All of this variables have the type index_t (aka unsigned int)
    or dim3 (struct of three index_t variables).
    """

    def generate_thread_count(self, generator: 'KernelGenerator'):
        if not generator.registered('thread_count'):
            generator.register('thread_count')

            # TODO find an algorithm for this instead of hard coding
            # for d in range(self.stencil.d_):
            #    # dependencies
            #    self.dimension(gen, d)

            # we target 512 threads at the moment
            if 1 == generator.stencil.stencil.d():
                generator.append_launcher_buffer("const auto thread_count = dim3{16u};")
            if 2 == generator.stencil.stencil.d():
                generator.append_launcher_buffer("const auto thread_count = dim3{16u, 16u};")
            if 3 == generator.stencil.stencil.d():
                generator.append_launcher_buffer("const auto thread_count = dim3{8u, 8u, 8u};")

    def generate_block_count(self, generator: 'KernelGenerator'):
        if not generator.registered('block_count'):
            generator.register('block_count')

            # dependencies
            self.generate_thread_count(generator)
            for d in range(generator.stencil.stencil.d()):
                self.generate_dimension(generator, d, hook_into_kernel=False)

            # generate
            coord = {0: 'x', 1: 'y', 2: 'z'}

            generator.append_launcher_buffer('')
            for d in range(generator.stencil.stencil.d()):
                generator.append_launcher_buffer(f"assert((dimension{d} % thread_count.{coord[d]}) == 0u);")

            dimensions = ', '.join([f"dimension{d} / thread_count.{coord[d]}" for d in range(generator.stencil.stencil.d())])

            generator.append_launcher_buffer(f"const auto block_count = dim3{{{dimensions}}};")
            generator.append_launcher_buffer('')

    def generate_index(self, generator: 'KernelGenerator', d: int):
        if not generator.registered(f"index{d}"):
            generator.register(f"index{d}")

            # generate
            coord = {0: 'x', 1: 'y', 2: 'z'}

            generator.append_index_buffer(f"const index_t index{d} = blockIdx.{coord[d]} * blockDim.{coord[d]} + threadIdx.{coord[d]};")

    def generate_dimension(self, generator: 'KernelGenerator', d: int, hook_into_kernel: bool):
        if not generator.registered(('dimension', d)):
            generator.register(('dimension', d))

            generator.append_launcher_buffer(f"const auto dimension{d} = static_cast<index_t> (f.sizes()[{d + 1}]);")

        if hook_into_kernel and not generator.kernel_hooked(('dimension', d)):
            generator.kernel_hook(('dimension', d), f"const index_t dimension{d}", f"dimension{d}")

    def generate_length(self, generator: 'KernelGenerator', d: int, hook_into_kernel: bool):
        if d == 0:  # length0 is an alias

            if not generator.registered(('length', 0, hook_into_kernel)):
                generator.register(('length', 0, hook_into_kernel))

                # dependencies
                self.generate_dimension(generator, 0, hook_into_kernel=hook_into_kernel)

                # generation
                if hook_into_kernel:
                    generator.append_index_buffer('const index_t &length0 = dimension0;')
                else:
                    generator.append_launcher_buffer('const index_t &length0 = dimension0;')

        else:

            if not generator.registered(('length', d)):
                generator.register(('length', d))

                # dependencies
                self.generate_dimension(generator, d, hook_into_kernel=False)
                self.generate_length(generator, d - 1, hook_into_kernel=False)

                # generation
                generator.append_launcher_buffer(f"const index_t length{d} = dimension{d} * length{d - 1};")

            if hook_into_kernel and not generator.kernel_hooked(('length', d)):
                generator.kernel_hook(('length', d), f"const index_t length{d}", f"length{d}")

    def generate_offset(self, generator: 'KernelGenerator'):
        if not generator.registered('offset'):
            generator.register('offset')

            # dependencies
            self.generate_index(generator, 0)
            for d in range(1, generator.stencil.stencil.d()):
                self.generate_index(generator, d)
                self.generate_length(generator, d - 1, hook_into_kernel=True)

            # generate
            offsets = ['(index0)']
            for d in range(1, generator.stencil.stencil.d()):
                offsets.append(f"(index{d} * length{d - 1})")

            generator.append_index_buffer(f"const index_t offset = {' + '.join(offsets)};")