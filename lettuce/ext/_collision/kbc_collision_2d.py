from .. import Force
from ... import Flow, Collision

__all__ = ['KBCCollision2D']


class KBCCollision2D(Collision):
    """
    Entropic multi-relaxation time model according to Karlin et al. in two dimensions
    """

    def __init__(self, lattice, tau):
        Collision.__init__(self, lattice)
        self.lattice = lattice
        assert lattice.Q == 9, LettuceException("KBC2D only realized for D2Q9")
        self.tau = tau
        self.beta = 1. / (2 * tau)

        # Build a matrix that contains the indices
        self.M = torch.zeros([3, 3, 9], device=lattice.device, dtype=lattice.dtype)
        for i in range(3):
            for j in range(3):
                self.M[i, j] = lattice.e[:, 0] ** i * lattice.e[:, 1] ** j

    def kbc_moment_transform(self, f):
        """Transforms the f into the KBC moment representation"""
        m = torch.einsum('abq,qmn', self.M, f)
        rho = m[0, 0]
        m = m / rho
        m[0, 0] = rho

        return m

    def compute_s_seq_from_m(self, f, m):
        s = torch.zeros_like(f)

        T = m[2, 0] + m[0, 2]
        N = m[2, 0] - m[0, 2]

        Pi_xy = m[1, 1]

        s[0] = m[0, 0] * -T
        s[1] = 1. / 2. * m[0, 0] * (0.5 * (T + N))
        s[2] = 1. / 2. * m[0, 0] * (0.5 * (T - N))
        s[3] = 1. / 2. * m[0, 0] * (0.5 * (T + N))
        s[4] = 1. / 2. * m[0, 0] * (0.5 * (T - N))
        s[5] = 1. / 4. * m[0, 0] * (Pi_xy)
        s[6] = -s[5]
        s[7] = 1. / 4 * m[0, 0] * Pi_xy
        s[8] = -s[7]

        return s

    def __call__(self, f):
        # the deletes are not part of the algorithm, they just keep the memory usage lower
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        # k = torch.zeros_like(f)

        m = self.kbc_moment_transform(f)
        delta_s = self.compute_s_seq_from_m(f, m)

        # k[0] = m[0, 0]
        # k[1] = m[0, 0] / 2. * m[1, 0]
        # k[2] = m[0, 0] / 2. * m[0, 1]
        # k[3] = -m[0, 0] / 2. * m[1, 0]
        # k[4] = -m[0, 0] / 2. * m[0, 1]
        # k[5] = 0
        # k[6] = 0
        # k[7] = 0
        # k[8] = 0

        m = self.kbc_moment_transform(feq)

        delta_s -= self.compute_s_seq_from_m(f, m)
        del m
        delta_h = f - feq - delta_s

        sum_s = self.lattice.rho(delta_s * delta_h / feq)
        sum_h = self.lattice.rho(delta_h * delta_h / feq)
        del feq
        gamma_stab = 1. / self.beta - (2 - 1. / self.beta) * sum_s / sum_h
        gamma_stab[gamma_stab < 1E-15] = 2.0
        gamma_stab[torch.isnan(gamma_stab)] = 2.0
        f = f - self.beta * (2 * delta_s + gamma_stab * delta_h)
        return f
