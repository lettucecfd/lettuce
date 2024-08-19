from . import Force
from lettuce.util import append_axes

__all__ = ['Guo']


class Guo(Force):

    def __init__(self, flow, tau, acceleration):
        self.flow = flow
        self.tau = tau
        self.acceleration = flow.context.convert_to_tensor(acceleration)

    def source_term(self, u):
        emu = append_axes(self.flow.torch_stencil.e,
                          self.flow.torch_stencil.d) - u
        eu = self.flow.einsum("ib,b->i", [self.flow.torch_stencil.e, u])
        eeu = self.flow.einsum("ia,i->ia", [self.flow.torch_stencil.e, eu])
        emu_eeu = (emu / (self.flow.torch_stencil.cs ** 2)
                   + eeu / (self.flow.torch_stencil.cs ** 4))
        emu_eeuF = self.flow.einsum("ia,a->i", [emu_eeu, self.acceleration])
        weemu_eeuF = (append_axes(self.flow.torch_stencil.w,
                                  self.flow.torch_stencil.d)
                      * emu_eeuF)
        return (1 - 1 / (2 * self.tau)) * weemu_eeuF

    def u_eq(self, flow: 'Flow'):
        return (self.ueq_scaling_factor
                * append_axes(self.acceleration, self.flow.torch_stencil.d)
                / self.flow.rho())

    @property
    def ueq_scaling_factor(self):
        return 0.5

    def native_available(self) -> bool:
        return False

    def native_generator(self) -> 'NativeForce':
        pass
