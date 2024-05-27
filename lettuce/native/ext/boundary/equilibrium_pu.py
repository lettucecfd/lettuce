from ... import NativeBoundary
from ... import Generator

__all__ = ['NativeEquilibriumBoundaryPu']


class NativeEquilibriumBoundaryPu(NativeBoundary):

    def __init__(self, index):
        NativeBoundary.__init__(self, index)

    @staticmethod
    def create(index):
        return NativeEquilibriumBoundaryPu(index)


    def generate_velocity(self, generator: 'Generator'):
        if not generator.launcher_hooked(f"velocity{self.index}"):
            generator.launcher_hook(f"velocity{self.index}",f"at::Tensor velocity{self.index}", f"velocity{self.index}", f"simulation.flow.units.convert_velocity_to_lu(simulation.boundaries[{self.index}].velocity)")
        if not generator.kernel_hooked(f"velocity{self.index}"):
            generator.kernel_hook(f"velocity{self.index}", f"scalar_t* p_velocity{self.index}", f"velocity{self.index}.data<scalar_t>()")
            generator.append_global_buffer(f"const scalar_t* velocity{self.index}=&p_velocity{self.index}[node_index*d];")

    def generate_density(self, generator: 'Generator'):
        if not generator.launcher_hooked(f"density{self.index}"):
            generator.launcher_hook(f"density{self.index}",f"at::Tensor density{self.index}", f"density{self.index}", f"simulation.flow.units.convert_pressure_pu_to_density_lu(simulation.boundaries[{self.index}].pressure)")
        if not generator.kernel_hooked(f"density{self.index}"):
            generator.kernel_hook(f"density{self.index}", f"scalar_t* p_density{self.index}", f"density{self.index}.data<scalar_t>()")
            generator.append_global_buffer(f"const scalar_t density{self.index}=p_density{self.index}[node_index];")
    def generate(self, generator: 'Generator'):
        self.generate_velocity(generator)
        self.generate_density(generator)
        generator.equilibrium.generate_f_eq(generator, rho=f"density{self.index}", u=f"velocity{self.index}")
        # generator.equilibrium.generate_f_eq(generator)
        f_eq = f"f_eq_density{self.index}_velocity{self.index}"
        generator.append_pipeline_buffer(f"if (no_collision_mask[node_index] == {self.index})")
        generator.append_pipeline_buffer('{                      ')

        for i in range(generator.stencil.q):
            generator.append_pipeline_buffer(f"  f_reg[{i}] = {f_eq}[{i}];")

        generator.append_pipeline_buffer('}')
