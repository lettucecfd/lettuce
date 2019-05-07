"""
Collision models and related functions
"""


class BGKCollision(object):
    def __init__(self, lattice, tau):
        self.lattice = lattice
        self.tau = tau

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.quadratic_equilibrium(rho, u)
        f = f - 1.0/self.tau * (f-feq)
        return f




