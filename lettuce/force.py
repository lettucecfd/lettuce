
import torch
class Guo:
    def __init__(self, lattice, tau, F):
        self.lattice = lattice
        self.tau = tau
        self.F = F

    def source_term(self, f, u):
        index = [Ellipsis] + [None] * self.lattice.D
        emu= self.lattice.e[index] - u
        eu = self.lattice.einsum("ib,b->i", [self.lattice.e, u])
        eeu = self.lattice.einsum("ia,i->ia", [self.lattice.e, eu])
        emu_eeu= emu/(self.lattice.cs ** 2) + eeu/(self.lattice.cs ** 4)
        emu_eeuF = self.lattice.einsum("ia,a->i", [emu_eeu, self.F])
        weemu_eeuF = self.lattice.w[index] * emu_eeuF
        return (1-1/(2*self.tau)) * weemu_eeuF

    def u_eq(self, f, force):
        index = [Ellipsis] + [None] * self.lattice.D
        return force.A * force.F[index] / self.lattice.rho(f)

    @property
    def A(self):
        return 0.5

class ShanChen:
    def __init__(self, lattice, tau, F):
        self.lattice = lattice
        self.tau = tau
        self.F = F

    def source_term(self, f, u):
        return 0

    def u_eq(self, f, force):
        index = [Ellipsis] + [None] * self.lattice.D
        return force.A * force.F[index] / self.lattice.rho(f)

    @property
    def A(self):
        return self.tau*1