from typing import Optional

from . import *


class NativeCollision(NativeLatticeBase):
    _name = 'invalid'

    equilibrium: Optional[NativeEquilibrium]
    support_no_collision_mask: bool

    # noinspection PyShadowingNames
    def __init__(self, equilibrium: NativeEquilibrium = None, support_no_collision_mask=False):
        self.equilibrium = equilibrium if equilibrium is not None else NativeQuadraticEquilibrium()
        self.support_no_collision_mask = support_no_collision_mask

    @property
    def name(self):
        equilibrium_name = self.equilibrium.name if self.equilibrium is not None else ''
        mask_name = 'M' if self.support_no_collision_mask else ''
        return f"{self._name}{equilibrium_name}{mask_name}"

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask):
        raise NotImplementedError()

    def generate_no_collision_mask(self, generator: 'Generator'):
        if not generator.launcher_hooked('no_collision_mask'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation, 'no_collision_mask')")
            generator.launcher_hook('no_collision_mask', 'const at::Tensor no_collision_mask',
                                    'no_collision_mask', 'simulation.no_collision_mask')
        if not generator.kernel_hooked('no_collision_mask'):
            generator.kernel_hook('no_collision_mask', 'const byte_t* no_collision_mask',
                                  'no_collision_mask.data<byte_t>()')

    def generate_collision(self, generator: 'Generator'):
        raise NotImplementedError()


class NativeNoCollision(NativeCollision):
    _name = 'no'

    def __init__(self):
        NativeCollision.__init__(self)

    @property
    def name(self):
        return self._name

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        return NativeBGKCollision()

    def generate_collision(self, generator: 'Generator'):
        if not generator.registered('collision()'):
            generator.register('collision()')


class NativeBGKCollision(NativeCollision):
    _name = 'bgk'

    def __init__(self, equilibrium: NativeEquilibrium = None, support_no_collision_mask=False):
        NativeCollision.__init__(self, equilibrium, support_no_collision_mask)

    @staticmethod
    def create(equilibrium: NativeEquilibrium, support_no_collision_mask: bool):
        return NativeBGKCollision(equilibrium, support_no_collision_mask)

    def generate_tau_inv(self, generator: 'Generator'):
        if not generator.launcher_hooked('tau_inv'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation.collision, 'tau')")
            generator.launcher_hook('tau_inv', 'const double tau_inv', 'tau_inv', '1./simulation.collision.tau')
        if not generator.kernel_hooked('tau_inv'):
            generator.kernel_hook('tau_inv', 'const scalar_t tau_inv', 'static_cast<scalar_t>(tau_inv)')

    def generate_collision(self, generator: 'Generator'):
        if generator.registered('collide()'):
            return

        generator.register('collide()')

        # dependencies

        if self.support_no_collision_mask:
            self.generate_no_collision_mask(generator)

        self.generate_tau_inv(generator)
        self.equilibrium.generate_f_eq(generator)

        d = generator.stencil.stencil.D()

        # generate
        if self.support_no_collision_mask:
            coord = generator.lattice.get_mask_coordinate(generator, ['index[0]', 'index[1]', 'index[2]'][:d])
            generator.append_index_buffer(f"if(!no_collision_mask[{coord}])")

        generator.append_distribution_buffer('f_reg[i] = f_reg[i] - (tau_inv * (f_reg[i] - f_eq));')
