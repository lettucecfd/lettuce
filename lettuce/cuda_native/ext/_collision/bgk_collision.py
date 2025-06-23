from typing import Optional

from lettuce.cuda_native import DefaultCodeGeneration, Parameter
from ... import NativeCollision

__all__ = ['NativeBGKCollision']


class NativeBGKCollision(NativeCollision):

    def __init__(self, index: int, force: Optional['NativeForce'] = None):
        NativeCollision.__init__(self, index)
        self.force = force

    @staticmethod
    def create(index: int, force: Optional['NativeForce'] = None):
        if force is None:
            return None
        return NativeBGKCollision(index)

    def cuda_tau_inv(self, reg: 'DefaultCodeGeneration'):
        variable = f"tau_inv{hex(id(self))[2:]}"
        return reg.cuda_hook('1.0 / simulation.collision.tau', Parameter('double', variable))

    def kernel_tau_inv(self, reg: 'DefaultCodeGeneration'):
        variable = self.cuda_tau_inv(reg)
        return reg.kernel_hook(f"static_cast<scalar_t>({variable})", Parameter('scalar_t', variable))

    def generate(self, reg: 'DefaultCodeGeneration'):
        tau_inv = self.kernel_tau_inv(reg)

        for q in range(reg.stencil.q):
            f_q = reg.f_reg(q)
            f_eq_q = reg.equilibrium.f_eq(reg, q)
            reg.pipe.append(f"{f_q}={f_q}-({tau_inv}*({f_q}-{f_eq_q}));")
