from .. import Force
from ... import Flow, Collision

__all__ = ['SmagorinskyCollision']


class SmagorinskyCollision(Collision):
    """
    Smagorinsky large eddy simulation (LES) _collision model with BGK operator.
    """

    def __init__(self, lattice, tau, smagorinsky_constant=0.17, force=None):
        Collision.__init__(self, lattice)
        self.force = force
        self.lattice = lattice
        self.tau = tau
        self.iterations = 2
        self.tau_eff = tau
        self.constant = smagorinsky_constant

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u_eq = 0 if self.force is None else self.force.u_eq(f)
        u = self.lattice.u(f) + u_eq
        feq = self.lattice.equilibrium(rho, u)
        S_shear = self.lattice.shear_tensor(f - feq)
        S_shear /= (2.0 * rho * self.lattice.cs ** 2)
        self.tau_eff = self.tau
        nu = (self.tau - 0.5) / 3.0

        for i in range(self.iterations):
            S = S_shear / self.tau_eff
            S = self.lattice.einsum('ab,ab->', [S, S])
            nu_t = self.constant ** 2 * S
            nu_eff = nu + nu_t
            self.tau_eff = nu_eff * 3.0 + 0.5
        Si = 0 if self.force is None else self.force.source_term(u)
        return f - 1.0 / self.tau_eff * (f - feq) + Si
