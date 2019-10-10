

class Equilibrium():
    pass


class QuadraticEquilibrium(Equilibrium):
    def __init__(self, lattice):
        self.lattice = lattice

    def __call__(self, rho, u, *args):
        exu = self.lattice.einsum("qd,d->q", [self.lattice.e, u])
        uxu = self.lattice.einsum("d,d->", [u,u])
        feq = self.lattice.einsum(
            "q,q->q",
            [self.lattice.w,
             rho * ((2 * exu - uxu) / (2 * self.lattice.cs ** 2) + 0.5 * (exu / (self.lattice.cs ** 2)) ** 2 + 1)]
        )
        return feq


class IncompressibleQuadraticEquilibrium(Equilibrium):
    def __init__(self, lattice, rho0=1.0):
        self.lattice = lattice
        self.rho0 = rho0

    def __call__(self, rho, u, *args):
        exu = self.lattice.einsum("qd,d->q", [self.lattice.e, u])
        uxu = self.lattice.einsum("d,d->", [u,u])
        feq = self.lattice.einsum(
            "q,q->q",
            [self.lattice.w,
             rho + self.rho0*((2 * exu - uxu) / (2 * self.lattice.cs ** 2) + 0.5 * (exu / (self.lattice.cs ** 2)) ** 2)
             ]
        )
        return feq
