from . import Force
from lettuce.util import append_axes

__all__ = ['ShanChen']


class ShanChen(Force):

    def __init__(self, flow, tau, acceleration):
        self.tau = tau
        self.acceleration = flow.context.convert_to_tensor(acceleration)

    def source_term(self, u):
        return 0

    def u_eq(self, flow: 'Flow'):
        return (self.ueq_scaling_factor
                * append_axes(self.acceleration, flow.stencil.d)
                / flow.rho())

    @property
    def ueq_scaling_factor(self):
        return self.tau * 1

    def native_available(self) -> bool:
        return False

    def native_generator(self) -> 'NativeForce':
        pass
