
__all__ = ["Guo", "ShanChen"]

from .util import append_axes


class Guo:
    def __init__(self, lattice, tau, acceleration):
        self.lattice = lattice
        self.tau = tau
        self.acceleration = lattice.convert_to_tensor(acceleration)

    def source_term(self, u):
        emu = append_axes(self.lattice.e, self.lattice.D) - u
        eu = self.lattice.einsum("ib,b->i", [self.lattice.e, u])
        eeu = self.lattice.einsum("ia,i->ia", [self.lattice.e, eu])
        emu_eeu = emu / (self.lattice.cs ** 2) + eeu / (self.lattice.cs ** 4)
        emu_eeuF = self.lattice.einsum("ia,a->i", [emu_eeu, self.acceleration])
        weemu_eeuF = append_axes(self.lattice.w, self.lattice.D) * emu_eeuF
        return (1 - 1 / (2 * self.tau)) * weemu_eeuF

    def u_eq(self, f):
        return self.ueq_scaling_factor * append_axes(self.acceleration, self.lattice.D) / self.lattice.rho(f)

    @property
    def ueq_scaling_factor(self):
        return 0.5


class ShanChen:
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
