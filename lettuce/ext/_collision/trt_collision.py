from .. import Force
from ... import Flow, Collision

__all__ = ['TRTCollision']


class TRTCollision(Collision):
    """
    Two relaxation time _collision model - standard implementation (cf. Krüger 2017)
    """

    def __init__(self, lattice, tau, tau_minus=1.0):
        Collision.__init__(self, lattice)
        self.lattice = lattice
        self.tau_plus = tau
        self.tau_minus = tau_minus

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        f_diff_neq = ((f + f[self.lattice.stencil.opposite]) - (feq + feq[self.lattice.stencil.opposite])) / (
                2.0 * self.tau_plus)
        f_diff_neq += ((f - f[self.lattice.stencil.opposite]) - (feq - feq[self.lattice.stencil.opposite])) / (
                2.0 * self.tau_minus)
        f = f - f_diff_neq
        return f
