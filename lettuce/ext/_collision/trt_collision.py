from ... import Flow, Collision

__all__ = ['TRTCollision']


class TRTCollision(Collision):
    """
    Two relaxation time collision model - standard implementation (cf. Krüger 2017)
    """

    def __init__(self, tau, tau_minus=1.0):
        Collision.__init__(self)
        self.tau_plus = tau
        self.tau_minus = tau_minus

    def __call__(self, flow: 'Flow'):
        # rho = flow.rho()
        # u = flow.u()
        feq = flow.equilibrium(flow)
        f_diff_neq = (((flow.f + flow.f[flow.stencil.opposite])
                      - (feq + feq[flow.stencil.opposite]))
                      / (2.0 * self.tau_plus))
        f_diff_neq += (((flow.f - flow.f[flow.stencil.opposite])
                       - (feq - feq[flow.stencil.opposite]))
                       / (2.0 * self.tau_minus))
        f = flow.f - f_diff_neq
        return f

    def native_available(self) -> bool:
        return False

    def native_generator(self) -> 'NativeCollision':
        pass
