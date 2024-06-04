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
            generator.launcher_hook(f"velocity{self.index}", f"at::Tensor velocity{self.index}", f"velocity{self.index}",
                                    f"simulation.flow.units.convert_velocity_to_lu(simulation.boundaries[{self.index}].velocity)")
        if not generator.kernel_hooked(f"velocity{self.index}"):
            generator.kernel_hook(f"velocity{self.index}", f"scalar_t* p_velocity{self.index}", f"velocity{self.index}.data<scalar_t>()")

            generator.kernel_hook(f"velocity{self.index}dimension0", f"index_t velocity{self.index}dimension0", f"velocity{self.index}dimensions[0]")
            generator.kernel_hook(f"velocity{self.index}dimension1", f"index_t velocity{self.index}dimension1", f"velocity{self.index}dimensions[1]", cond=generator.stencil.d > 1)
            generator.kernel_hook(f"velocity{self.index}dimension2", f"index_t velocity{self.index}dimension2", f"velocity{self.index}dimensions[2]", cond=generator.stencil.d > 2)

            generator.append_launcher_buffer(f"  index_t velocity{self.index}dimensions[d] = {{   ")
            generator.append_launcher_buffer(f"      velocity{self.index}.sizes()[1] > 1 ? 1 : 0  ")
            generator.append_launcher_buffer(f"    , velocity{self.index}.sizes()[2] > 1 ? 1 : 0  ", cond=generator.stencil.d > 1)
            generator.append_launcher_buffer(f"    , velocity{self.index}.sizes()[3] > 1 ? 1 : 0  ", cond=generator.stencil.d > 2)
            generator.append_launcher_buffer(f"  }};                                              ")

            generator.append_global_buffer(f"scalar_t* velocity{self.index};                                                ")
            generator.append_global_buffer(f"{{                                                                             ")
            generator.append_global_buffer(f"  index_t velocity{self.index}dimensions[d] = {{                               ")
            generator.append_global_buffer(f"      velocity{self.index}dimension0                                           ")
            generator.append_global_buffer(f"    , velocity{self.index}dimension1                                           ", cond=generator.stencil.d > 1)
            generator.append_global_buffer(f"    , velocity{self.index}dimension2                                           ", cond=generator.stencil.d > 2)
            generator.append_global_buffer(f"  }};                                                                          ")
            generator.append_global_buffer(f"  index_t velocity{self.index}index = 0;                                       ")
            generator.append_global_buffer(f"  index_t velocity{self.index}multiplier = 1;                                  ")
            generator.append_global_buffer(f"  for (index_t i = d - 1; i >= 0; --i) {{                                      ")
            generator.append_global_buffer(f"    if (velocity{self.index}dimensions[i]) {{                                  ")
            generator.append_global_buffer(f"      velocity{self.index}index += index[i] * velocity{self.index}multiplier;  ")
            generator.append_global_buffer(f"      velocity{self.index}multiplier *= dimension[i];                          ")
            generator.append_global_buffer(f"    }}                                                                         ")
            generator.append_global_buffer(f"  }}                                                                           ")
            generator.append_global_buffer(f"  velocity{self.index}=&p_velocity{self.index}[velocity{self.index}index*d];   ")
            generator.append_global_buffer(f"}}                                                                             ")

    def generate_density(self, generator: 'Generator'):
        if not generator.launcher_hooked(f"density{self.index}"):
            generator.launcher_hook(f"density{self.index}", f"at::Tensor density{self.index}", f"density{self.index}",
                                    f"simulation.flow.units.convert_pressure_pu_to_density_lu(simulation.boundaries[{self.index}].pressure)")
        if not generator.kernel_hooked(f"density{self.index}"):
            generator.kernel_hook(f"density{self.index}", f"scalar_t* p_density{self.index}", f"density{self.index}.data<scalar_t>()")

            generator.kernel_hook(f"density{self.index}dimension0", f"index_t density{self.index}dimension0", f"density{self.index}dimensions[0]")
            generator.kernel_hook(f"density{self.index}dimension1", f"index_t density{self.index}dimension1", f"density{self.index}dimensions[1]", cond=generator.stencil.d > 1)
            generator.kernel_hook(f"density{self.index}dimension2", f"index_t density{self.index}dimension2", f"density{self.index}dimensions[2]", cond=generator.stencil.d > 2)

            generator.append_launcher_buffer(f"  index_t density{self.index}dimensions[d] = {{   ")
            generator.append_launcher_buffer(f"      density{self.index}.sizes()[1] > 1 ? 1 : 0  ")
            generator.append_launcher_buffer(f"    , density{self.index}.sizes()[2] > 1 ? 1 : 0  ", cond=generator.stencil.d > 1)
            generator.append_launcher_buffer(f"    , density{self.index}.sizes()[3] > 1 ? 1 : 0  ", cond=generator.stencil.d > 2)
            generator.append_launcher_buffer(f"  }};                                             ")

            generator.append_global_buffer(f"scalar_t density{self.index};                                                ")
            generator.append_global_buffer(f"{{                                                                           ")
            generator.append_global_buffer(f"  index_t density{self.index}dimensions[d] = {{                              ")
            generator.append_global_buffer(f"      density{self.index}dimension0                                          ")
            generator.append_global_buffer(f"    , density{self.index}dimension1                                          ", cond=generator.stencil.d > 1)
            generator.append_global_buffer(f"    , density{self.index}dimension2                                          ", cond=generator.stencil.d > 2)
            generator.append_global_buffer(f"  }};                                                                        ")
            generator.append_global_buffer(f"  index_t density{self.index}index = 0;                                      ")
            generator.append_global_buffer(f"  index_t density{self.index}multiplier = 1;                                 ")
            generator.append_global_buffer(f"  for (index_t i = d - 1; i >= 0; --i) {{                                    ")
            generator.append_global_buffer(f"    if (density{self.index}dimensions[i]) {{                                 ")
            generator.append_global_buffer(f"      density{self.index}index += index[i] * density{self.index}multiplier;  ")
            generator.append_global_buffer(f"      density{self.index}multiplier *= dimension[i];                         ")
            generator.append_global_buffer(f"    }}                                                                       ")
            generator.append_global_buffer(f"  }}                                                                         ")
            generator.append_global_buffer(f"  density{self.index}=p_density{self.index}[density{self.index}index];       ")
            generator.append_global_buffer(f"}}                                                                           ")

    def generate(self, generator: 'Generator'):
        self.generate_velocity(generator)
        self.generate_density(generator)
        generator.equilibrium.generate_f_eq(generator, rho=f"density{self.index}", u=f"velocity{self.index}")
        # generator._equilibrium.generate_f_eq(generator)
        f_eq = f"f_eq_density{self.index}_velocity{self.index}"
        generator.append_pipeline_buffer(f"if (no_collision_mask[node_index] == {self.index})")
        generator.append_pipeline_buffer('{                      ')

        for i in range(generator.stencil.q):
            generator.append_pipeline_buffer(f"  f_reg[{i}] = {f_eq}[{i}];")

        generator.append_pipeline_buffer('}')
