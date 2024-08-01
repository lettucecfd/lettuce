from typing import Optional

from ... import NativeCollision, NativeEquilibrium

__all__ = ['NativeBGKCollision']


class NativeBGKCollision(NativeCollision):

    def __init__(self, force: Optional['NativeForce'] = None):
        NativeCollision.__init__(self)
        self.force = force

    @staticmethod
    def create(force: Optional['NativeForce'] = None):
        if force is None:
            return None
        return NativeBGKCollision()

    # noinspection PyMethodMayBeStatic
    def generate_tau_inv(self, generator: 'Generator'):
        tau_inv = f"tau_inv{hex(id(self))[2:]}"

        if not generator.launcher_hooked(tau_inv):
            buffer = generator.append_python_wrapper_before_buffer

            buffer('assert hasattr(simulation, \'collision\')')
            buffer('assert hasattr(simulation.collision, \'tau\')')

            generator.launcher_hook(tau_inv, f"double {tau_inv}", tau_inv, '1./simulation.collision.tau')

        if not generator.kernel_hooked(tau_inv):
            generator.kernel_hook(tau_inv, f"scalar_t {tau_inv}", f"static_cast<scalar_t>({tau_inv})")

        return tau_inv

    def generate(self, generator: 'Generator'):
        tau_inv = self.generate_tau_inv(generator)
        generator.equilibrium.generate_f_eq(generator)

        buffer = generator.append_pipeline_buffer
        ncm = generator.support_no_collision_mask

        buffer('if(no_collision_mask[node_index] == 0) {                       ', cond=ncm)
        buffer('#pragma unroll                                                 ')
        buffer('  for (index_t i = 0; i < q; ++i)                              ')
        buffer(f"   f_reg[i] = f_reg[i] - ({tau_inv} * (f_reg[i] - f_eq[i]));  ")
        buffer('}                                                              ', cond=ncm)