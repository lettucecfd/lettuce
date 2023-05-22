from abc import abstractmethod, ABC

from . import *


class NativeCollision(NativePipelineStep, ABC):

    @staticmethod
    @abstractmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask):
        ...

    # noinspection PyMethodMayBeStatic
    def generate_no_collision_mask(self, generator: 'Generator'):
        if not generator.launcher_hooked('no_collision_mask'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation, 'no_collision_mask')")
            generator.launcher_hook('no_collision_mask', 'const at::Tensor no_collision_mask',
                                    'no_collision_mask', 'simulation.no_collision_mask')
        if not generator.kernel_hooked('no_collision_mask'):
            generator.kernel_hook('no_collision_mask', 'const byte_t* no_collision_mask',
                                  'no_collision_mask.data<byte_t>()')


class NativeNoCollision(NativeCollision):
    def __init__(self):
        NativeCollision.__init__(self)

    @property
    def name(self) -> str:
        return 'NoCollision'

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        assert equilibrium is None, "NativeNoCollision does not support equilibrium"
        assert not support_no_collision_mask, "NativeNoCollision does not support no_collision_mask"
        return NativeNoCollision()

    def generate(self, generator: 'Generator'):
        if not generator.registered('collision()'):
            generator.register('collision()')


class NativeBGKCollision(NativeCollision):

    def __init__(self, equilibrium: NativeEquilibrium, support_no_collision_mask=False):
        NativeCollision.__init__(self)
        self.equilibrium = equilibrium
        self.support_no_collision_mask = support_no_collision_mask

    @property
    def name(self) -> str:
        equilibrium_name = self.equilibrium.name
        mask_name = 'Masked' if self.support_no_collision_mask else ''
        return f"{mask_name}BGKCollision{equilibrium_name}"

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        return NativeBGKCollision(equilibrium, support_no_collision_mask)

    # noinspection PyMethodMayBeStatic
    def generate_tau_inv(self, generator: 'Generator'):
        if not generator.launcher_hooked('tau_inv'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation.collision, 'tau')")
            generator.launcher_hook('tau_inv', 'double tau_inv', 'tau_inv', '1./simulation.collision.tau')
        if not generator.kernel_hooked('tau_inv'):
            generator.kernel_hook('tau_inv', 'scalar_t tau_inv', 'static_cast<scalar_t>(tau_inv)')

    def generate(self, generator: 'Generator'):
        # dependencies
        generator.read.generate_f_reg(generator)

        if self.support_no_collision_mask:
            generator.cuda.generate_index(generator)
            self.generate_no_collision_mask(generator)

        self.generate_tau_inv(generator)
        self.equilibrium.generate_f_eq(generator)

        # generate
        if self.support_no_collision_mask:
            d = generator.stencil.stencil.D()
            coord = generator.lattice.get_mask_coordinate(generator, ['index[0]', 'index[1]', 'index[2]'][:d])
            generator.append_global_buffer(f"if(!no_collision_mask[{coord}])")

        pipe_buf = generator.append_pipeline_buffer
        pipe_buf('  # pragma unroll                                           ')
        pipe_buf('  for (index_t i = 0; i < q; ++i)                           ')
        pipe_buf('  {                                                         ')
        pipe_buf('    f_reg[i] = f_reg[i] - (tau_inv * (f_reg[i] - f_eq[i])); ')
        pipe_buf('  }                                                         ')
