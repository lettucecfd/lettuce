from typing import Optional

from ... import NativeCollision, NativeEquilibrium

__all__ = ['NativeBGKCollision']


class NativeBGKCollision(NativeCollision):

    def __init__(self, force: Optional['NativeForce'] = None):
        NativeCollision.__init__(self)
        self.force = force

    @staticmethod
    def create(force: Optional['NativeForce'] = None):
        return NativeBGKCollision()

    # noinspection PyMethodMayBeStatic
    def generate_tau_inv(self, generator: 'Generator'):
        if not generator.launcher_hooked('tau_inv'):
            generator.append_python_wrapper_before_buffer("assert hasattr(simulation.collision, 'tau')")
            generator.launcher_hook('tau_inv', 'double tau_inv', 'tau_inv', '1./simulation.collision.tau')
        if not generator.kernel_hooked('tau_inv'):
            generator.kernel_hook('tau_inv', 'scalar_t tau_inv', 'static_cast<scalar_t>(tau_inv)')

    def generate(self, generator: 'Generator'):
        self.generate_tau_inv(generator)
        generator.equilibrium.generate_f_eq(generator)

        generator.append_pipeline_buffer('  if(no_collision_mask[node_index] == 0) {                     ', cond=generator.support_no_collision_mask)
        generator.append_pipeline_buffer('#pragma unroll                                                 ')
        generator.append_pipeline_buffer('    for (index_t i = 0; i < q; ++i)                            ')
        generator.append_pipeline_buffer('      f_reg[i] = f_reg[i] - (tau_inv * (f_reg[i] - f_eq[i]));  ')
        generator.append_pipeline_buffer('  }                                                            ', cond=generator.support_no_collision_mask)
