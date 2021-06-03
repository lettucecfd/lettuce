__all__ = ["Guo", "ShanChen"]


class Guo:
    def __init__(self, lattice, tau, acceleration):
        self.lattice = lattice
        self.tau = tau
        self.acceleration = lattice.convert_to_tensor(acceleration)

    def source_term(self, u):
        index = [Ellipsis] + [None] * self.lattice.D
        emu = self.lattice.e[index] - u
        eu = self.lattice.einsum("ib,b->i", [self.lattice.e, u])
        eeu = self.lattice.einsum("ia,i->ia", [self.lattice.e, eu])
        emu_eeu = emu / (self.lattice.cs ** 2) + eeu / (self.lattice.cs ** 4)
        emu_eeuF = self.lattice.einsum("ia,a->i", [emu_eeu, self.acceleration])
        weemu_eeuF = self.lattice.w[index] * emu_eeuF
        return (1 - 1 / (2 * self.tau)) * weemu_eeuF

    def u_eq(self, f):
        index = [Ellipsis] + [None] * self.lattice.D
        return self.ueq_scaling_factor * self.acceleration[index] / self.lattice.rho(f)

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
        index = [Ellipsis] + [None] * self.lattice.D
        return self.ueq_scaling_factor * self.acceleration[index] / self.lattice.rho(f)

    @property
    def ueq_scaling_factor(self):
        return self.tau * 1
