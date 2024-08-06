from ... import Flow, Collision
from .. import Force

__all__ = ['SmagorinskyCollision']


class SmagorinskyCollision(Collision):
    """
    Smagorinsky large eddy simulation (LES) collision model with BGK operator.
    """

    def __init__(self, tau, smagorinsky_constant=0.17, force: 'Force' = None):
        Collision.__init__(self)
        self.force = force
        self.tau = tau
        self.iterations = 2
        self.tau_eff = tau
        self.constant = smagorinsky_constant

    def __call__(self, flow: 'Flow'):
        rho = flow.rho()
        u_eq = 0 if self.force is None else self.force.u_eq(flow)
        u = flow.u() + u_eq
        feq = flow.equilibrium(flow, rho, u)
        S_shear = flow.shear_tensor(flow.f - feq)
        S_shear /= (2.0 * rho * flow.stencil.cs ** 2)
        self.tau_eff = self.tau
        nu = (self.tau - 0.5) / 3.0

        for i in range(self.iterations):
            S = S_shear / self.tau_eff
            S = flow.einsum('ab,ab->', [S, S])
            nu_t = self.constant ** 2 * S
            nu_eff = nu + nu_t
            self.tau_eff = nu_eff * 3.0 + 0.5
        Si = 0 if self.force is None else self.force.source_term(u)
        return flow.f - 1.0 / self.tau_eff * (flow.f - feq) + Si

    def native_available(self) -> bool:
        return False

    def native_generator(self) -> 'NativeCollision':
        pass
