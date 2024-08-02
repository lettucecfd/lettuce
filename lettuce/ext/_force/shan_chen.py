from . import Force
from lettuce.util import append_axes

__all__ = ['ShanChen']


class ShanChen(Force):
    def __init__(self, lattice, tau, acceleration):
        self.lattice = lattice
        self.tau = tau
        self.acceleration = lattice.convert_to_tensor(acceleration)

    def source_term(self, u):
        return 0

    def u_eq(self, f):
        return self.ueq_scaling_factor * append_axes(self.acceleration, self.lattice.D) / self.lattice.rho(f)

    @property
    def ueq_scaling_factor(self):
        return self.tau * 1
