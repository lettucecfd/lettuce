from lettuce.gencuda import KernelGenerator


class Cuda:
    """
    """

    def __init__(self):
        """
        """
        pass

    def thread_count(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('thread_count'):
            gen.register('thread_count')

            # TODO find an algorithm for this instead of hard coding
            # for d in range(self.stencil.d_):
            #    # dependencies
            #    self.dimension(gen, d)

            # we target 512 threads at the moment
            if 1 == gen.stencil.d_:
                gen.wrp("const auto thread_count = dim3{16u};")
            if 2 == gen.stencil.d_:
                gen.wrp("const auto thread_count = dim3{16u, 16u};")
            if 3 == gen.stencil.d_:
                gen.wrp("const auto thread_count = dim3{8u, 8u, 8u};")

    def block_count(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('block_count'):
            gen.register('block_count')

            # dependencies
            self.thread_count(gen)
            for d in range(gen.stencil.d_):
                self.dimension(gen, d, n=False)

            # generate
            coord = {0: 'x', 1: 'y', 2: 'z'}

            gen.wrp('')
            for d in range(gen.stencil.d_):
                gen.wrp(f"assert((dimension{d} % thread_count.{coord[d]}) == 0u);")

            dimensions = ', '.join([f"dimension{d} / thread_count.{coord[d]}" for d in range(gen.stencil.d_)])

            gen.wrp(f"const auto block_count = dim3{{{dimensions}}};")
            gen.wrp('')

    def index(self, gen: 'KernelGenerator', d: int):
        """
        """

        if not gen.registered(f"index{d}"):
            gen.register(f"index{d}")

            # generate
            coord = {0: 'x', 1: 'y', 2: 'z'}

            gen.idx(f"const index_t index{d} = blockIdx.{coord[d]} * blockDim.{coord[d]} + threadIdx.{coord[d]};")

    def dimension(self, gen: 'KernelGenerator', d: int, n: bool):
        """
        """

        if not gen.registered(('dimension', d)):
            gen.register(('dimension', d))

            gen.wrp(f"const auto dimension{d} = static_cast<index_t> (f.sizes()[{d + 1}]);")

        if n and not gen.kernel_hooked(f"dimension{d}"):
            gen.kernel_hook(f"dimension{d}", f"const index_t dimension{d}", f"dimension{d}")

    def length(self, gen: 'KernelGenerator', d: int, n: bool):
        """
        """

        if d == 0:  # length0 is an alias

            if not gen.registered(('length0', n)):
                gen.register(('length0', n))

                # dependencies
                self.dimension(gen, 0, n=n)

                # generation
                if n:
                    gen.idx('const index_t &length0 = dimension0;')
                else:
                    gen.wrp('const index_t &length0 = dimension0;')

        else:

            if not gen.registered(f"length{d}"):
                gen.register(f"length{d}")

                # dependencies
                self.dimension(gen, d, n=False)
                self.length(gen, d - 1, n=False)

                # generation
                gen.wrp(f"const index_t length{d} = dimension{d} * length{d - 1};")

            if n and not gen.kernel_hooked(f"length{d}"):
                gen.kernel_hook(f"length{d}", f"const index_t length{d}",
                                f"length{d}")

    def offset(self, gen: 'KernelGenerator'):
        """
        """

        if not gen.registered('offset'):
            gen.register('offset')

            # dependencies
            self.index(gen, 0)
            for d in range(1, gen.stencil.d_):
                self.index(gen, d)
                self.length(gen, d - 1, n=True)

            # generate
            offsets = ['(index0)']
            for d in range(1, gen.stencil.d_):
                offsets.append(f"(index{d} * length{d - 1})")

            gen.idx(f"const index_t offset = {' + '.join(offsets)};")
