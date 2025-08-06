from lettuce.cuda_native import DefaultCodeGeneration
from lettuce.cuda_native import NativeEquilibrium
from lettuce.cuda_native import lettuce_hash

__all__ = ['NativeQuadraticEquilibrium']


class NativeQuadraticEquilibrium(NativeEquilibrium):

    # noinspection PyMethodMayBeStatic
    def uxu(self, reg: 'DefaultCodeGeneration', u: [str] = None) -> str:
        u = u or [reg.kernel_u(d) for d in range(reg.stencil.d)]
        # create unique variable per u
        variable = f"uxu_{lettuce_hash(u)}"
        code = '+'.join([f"{u[d]}*{u[d]}" for d in range(reg.stencil.d)])
        return reg.pipes.variable('scalar_t', variable, code)

    # noinspection PyMethodMayBeStatic
    def exu(self, reg: 'DefaultCodeGeneration', q: int, u: [str] = None) -> str:
        u = u or [reg.kernel_u(d) for d in range(reg.stencil.d)]
        # create unique variable per u
        variable = f"exu_{q}_{lettuce_hash(u)}"
        code = '+'.join([f"{reg.e(q, d)}*{u[d]}" for d in range(reg.stencil.d)])
        return reg.pipes.variable('scalar_t', variable, code)

    # noinspection PyMethodMayBeStatic
    def cs_pow_two(self, _) -> str:
        return 'static_cast<scalar_t>(1.0 / 3.0)'

    # noinspection PyMethodMayBeStatic
    def two_cs_pow_two(self, _) -> str:
        return 'static_cast<scalar_t>(2.0 / 3.0)'

    def f_eq(self, reg: 'DefaultCodeGeneration', q: int, rho: str = None, u: [str] = None):
        rho = rho or reg.kernel_rho()
        u = u or [reg.kernel_u(d) for d in range(reg.stencil.d)]

        # dependencies
        uxu = self.uxu(reg, u)
        exu_q = self.exu(reg, q, u)
        cs_pow_two = self.cs_pow_two(reg)
        two_cs_pow_two = self.two_cs_pow_two(reg)

        h = lettuce_hash([rho] + u)
        variable_tmp = f"f_eq_{q}_{h}_tmp"
        variable = f"f_eq_{q}_{h}"

        reg.pipes.variable('scalar_t', variable_tmp, f"{exu_q}/{cs_pow_two}")
        code = f"{rho}*{reg.w(q)}*(({exu_q}+{exu_q}-{uxu})/{two_cs_pow_two}+0.5*{variable_tmp}*{variable_tmp}+1.0)"
        return reg.pipes.variable('scalar_t', variable, code)
