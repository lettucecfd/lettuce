from ... import NativeBoundary, DefaultCodeGeneration, Parameter, CodeRegistryList

__all__ = ['NativeEquilibriumBoundaryPu']


class NativeEquilibriumBoundaryPu(NativeBoundary):

    def __init__(self, index):
        NativeBoundary.__init__(self, index)

    @staticmethod
    def create(index):
        return NativeEquilibriumBoundaryPu(index)

    def cuda_velocity(self, reg: 'DefaultCodeGeneration'):
        py_value = f"simulation.flow.units.convert_velocity_to_lu(simulation.boundaries[{self.index}].velocity)"
        return reg.cuda_hook(py_value, Parameter('at::Tensor', f"velocity_{self.index}"))

    def cuda_velocity_size(self, reg: 'DefaultCodeGeneration', d: int):
        assert d in range(reg.stencil.d)
        variable = self.cuda_velocity(reg)
        return f"static_cast<index_t>({variable}.size({d}))"

    def kernel_velocity_size(self, reg: 'DefaultCodeGeneration', d: int):
        assert d in range(reg.stencil.d)
        variable = self.cuda_velocity_size(reg, d)
        return reg.kernel_hook(variable, Parameter('index_t', f"velocity_{self.index}_size_{'xyz'[d]}"))

    def kernel_velocity(self, reg: 'DefaultCodeGeneration', d: int):
        assert d in range(reg.stencil.d)

        variable = self.cuda_velocity(reg)
        p_variable = reg.kernel_hook(f"{variable}.data<scalar_t>()", Parameter('scalar_t*', f"p_{variable}"))

        if not reg.pipe.registered(variable):

            code = CodeRegistryList(reg.stencil.d)
            code.append('[&]{')
            variable_i = code.mutable('index_t', f"{variable}_index", '0')
            variable_m = code.mutable('index_t', f"{variable}_multiplier", '1')
            for d in reversed(range(reg.stencil.d)):
                code.append(f"if({self.kernel_velocity_size(reg, d)}) {{")
                code.append(f"  {variable_i}+={reg.kernel_index(d)}*{variable_m};")
                code.append(f"  {variable_m}*={reg.kernel_size(d + 1)};", cond=bool(d))
                code.append(f"}}")
            code.append(f"return &{p_variable}[{variable_i}*{reg.d()}];")
            code.append('}()')

            reg.pipes.variable('scalar_t*', variable, str(code))

        return f"{variable}[{d}]"

    def cuda_density(self, reg: 'DefaultCodeGeneration'):
        py_value = f"simulation.flow.units.convert_pressure_pu_to_density_lu(simulation.boundaries[{self.index}].pressure)"
        return reg.cuda_hook(py_value, Parameter('at::Tensor', f"density_{self.index}"))

    def cuda_density_size(self, reg: 'DefaultCodeGeneration', d: int):
        assert d in range(reg.stencil.d)
        variable = self.cuda_density(reg)
        return f"static_cast<index_t>({variable}.sizes()[{d}])"

    def kernel_density_size(self, reg: 'DefaultCodeGeneration', d: int):
        assert d in range(reg.stencil.d)
        variable = self.cuda_density_size(reg, d)
        return reg.kernel_hook(variable, Parameter('index_t', f"density_{self.index}_size_{d}"))

    def kernel_density(self, reg: 'DefaultCodeGeneration'):
        variable = self.cuda_density(reg)
        p_variable = reg.kernel_hook(f"{variable}.data<scalar_t>()", Parameter('scalar_t*', f"p_{variable}"))

        code = CodeRegistryList(reg.stencil.d)
        code.append('[&]{')
        variable_i = code.mutable('index_t', f"{variable}_index", '0')
        variable_m = code.mutable('index_t', f"{variable}_multiplier", '1')
        for d in reversed(range(reg.stencil.d)):
            code.append(f"if({self.kernel_density_size(reg, d)}) {{")
            code.append(f"  {variable_i}+={reg.kernel_index(d)}*{variable_m};")
            code.append(f"  {variable_m}*={reg.kernel_size(d + 1)};", cond=bool(d))
            code.append(f"}}")
        code.append(f"return {p_variable}[{variable_i}];")
        code.append('}()')

        return reg.pipes.variable('scalar_t', variable, str(code))

    def generate(self, reg: 'DefaultCodeGeneration'):

        velocity = [self.kernel_velocity(reg, d) for d in range(reg.stencil.d)]
        density = self.kernel_density(reg)

        for q in range(reg.stencil.q):
            f_reg_q = reg.f_reg(q)
            f_eq_q = reg.equilibrium.f_eq(reg, q, rho=density, u=velocity)
            reg.pipe.append(f"  {f_reg_q} = {f_eq_q};")
