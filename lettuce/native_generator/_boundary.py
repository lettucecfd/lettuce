from abc import abstractmethod, ABC

from . import *


class NativeBoundary(NativePipelineStep, ABC):

    @staticmethod
    @abstractmethod
    def create(support_no_boundary_mask: bool):
        ...

    # noinspection PyMethodMayBeStatic
    def generate_no_boundary_mask(self, generator: 'Generator'):
        if not generator.launcher_hooked('no_boundary_mask'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation, 'no_boundary_mask')")
            generator.launcher_hook('no_boundary_mask', 'const at::Tensor no_boundary_mask',
                                    'no_boundary_mask', 'simulation.no_boundary_mask')
        if not generator.kernel_hooked('no_boundary_mask'):
            generator.kernel_hook('no_boundary_mask', 'const byte_t* no_boundary_mask',
                                  'no_boundary_mask.data<byte_t>()')


class NativeNoBoundary(NativeBoundary):
    def __init__(self):
        NativeBoundary.__init__(self)

    @property
    def name(self) -> str:
        return 'NoBoundary'

    @staticmethod
    def create(support_no_boundary_mask: bool):
        assert not support_no_boundary_mask, "NativeNoBoundary does not support no_boundary_mask"
        return NativeNoBoundary()

    def generate(self, generator: 'Generator'):
        return


class NativeBounceBackBoundary(NativeBoundary):

    def __init__(self, support_no_boundary_mask=False):
        NativeBoundary.__init__(self)
        self.support_no_boundary_mask = support_no_boundary_mask

    @property
    def name(self) -> str:
        mask_name = 'Masked' if self.support_no_boundary_mask else ''
        return f"{mask_name}BounceBackBoundary"

    @staticmethod
    def create(support_no_boundary_mask: bool):
        return NativeBounceBackBoundary(support_no_boundary_mask)

    # noinspection PyMethodMayBeStatic
    def generate_tau_inv(self, generator: 'Generator'):
        if not generator.launcher_hooked('tau_inv'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation.Boundary, 'tau')")
            generator.launcher_hook('tau_inv', 'double tau_inv', 'tau_inv', '1./simulation.Boundary.tau')
        if not generator.kernel_hooked('tau_inv'):
            generator.kernel_hook('tau_inv', 'scalar_t tau_inv', 'static_cast<scalar_t>(tau_inv)')

    def generate(self, generator: 'Generator'):
        # dependencies
        generator.read.generate_f_reg(generator)

        if self.support_no_boundary_mask:
            generator.cuda.generate_index(generator)
            self.generate_no_boundary_mask(generator)

        self.generate_tau_inv(generator)

        # generate
        pipe_buf = generator.append_pipeline_buffer

        d = generator.stencil.stencil.D()
        coord = generator.lattice.get_mask_coordinate(generator, ['index[0]', 'index[1]', 'index[2]'][:d])
        pipe_buf(f"if(!no_boundary_mask[{coord}]) {{                          ", cond=self.support_no_boundary_mask)

        pipe_buf('  {                                                         ')
        pipe_buf('  scalar_t bounce[q];                                       ')

        for i in range(generator.stencil.stencil.Q()):
            pipe_buf(f"bounce[{i}] = f_reg[{generator.stencil.stencil.opposite[i]}];")

        for i in range(generator.stencil.stencil.Q()):
            pipe_buf(f"f_reg[{i}] = f_reg[{i}];")

        pipe_buf('  }                                                         ')
        pipe_buf('}                                                           ', cond=self.support_no_boundary_mask)
