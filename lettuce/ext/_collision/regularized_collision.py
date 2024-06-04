import torch

from .. import Force
from ... import Flow, Collision

__all__ = ['RegularizedCollision']


class RegularizedCollision(Collision):
    """
    Regularized LBM according to Jonas Latt and Bastien Chopard (2006)
    """

    def __init__(self, lattice, tau):
        Collision.__init__(self, lattice)
        self.lattice = lattice
        self.tau = tau
        self.Q_matrix = torch.zeros([lattice.Q, lattice.D, lattice.D], device=lattice.device, dtype=lattice.dtype)

        for a in range(lattice.Q):
            for b in range(lattice.D):
                for c in range(lattice.D):
                    self.Q_matrix[a, b, c] = lattice.e[a, b] * lattice.e[a, c]
                    if b == c:
                        self.Q_matrix[a, b, c] -= lattice.cs * lattice.cs

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        pi_neq = self.lattice.shear_tensor(f - feq)
        cs4 = self.lattice.cs ** 4

        pi_neq = self.lattice.einsum("qab,ab->q", [self.Q_matrix, pi_neq])
        pi_neq = self.lattice.einsum("q,q->q", [self.lattice.w, pi_neq])

        fi1 = pi_neq / (2 * cs4)
        f = feq + (1. - 1. / self.tau) * fi1

        return f
