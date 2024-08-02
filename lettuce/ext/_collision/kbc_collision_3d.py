import torch

from .. import Force
from ... import Flow, Collision

__all__ = ['KBCCollision3D']


class KBCCollision3D(Collision):
    """
    Entropic multi-relaxation time-relaxation time model according to Karlin et al. in three dimensions
    """

    def __init__(self, lattice, tau):
        Collision.__init__(self, lattice)
        self.lattice = lattice
        assert lattice.Q == 27, "KBC only realized for D3Q27"
        self.tau = tau
        self.beta = 1. / (2 * tau)

        # Build a matrix that contains the indices
        self.M = torch.zeros([3, 3, 3, 27], device=lattice.device, dtype=lattice.dtype)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.M[i, j, k] = lattice.e[:, 0] ** i * lattice.e[:, 1] ** j * lattice.e[:, 2] ** k

    def kbc_moment_transform(self, f):
        """Transforms the f into the KBC moment representation"""
        m = torch.einsum('abcq,qmno', self.M, f)
        rho = m[0, 0, 0]
        m = m / rho
        m[0, 0, 0] = rho

        return m

    def compute_s_seq_from_m(self, f, m):
        s = torch.zeros_like(f)

        T = m[2, 0, 0] + m[0, 2, 0] + m[0, 0, 2]
        N_xz = m[2, 0, 0] - m[0, 0, 2]
        N_yz = m[0, 2, 0] - m[0, 0, 2]
        Pi_xy = m[1, 1, 0]
        Pi_xz = m[1, 0, 1]
        Pi_yz = m[0, 1, 1]

        s[0] = m[0, 0, 0] * -T
        s[1] = 1. / 6. * m[0, 0, 0] * (2 * N_xz - N_yz + T)
        s[2] = s[1]
        s[3] = 1. / 6. * m[0, 0, 0] * (2 * N_yz - N_xz + T)
        s[4] = s[3]
        s[5] = 1. / 6. * m[0, 0, 0] * (-N_xz - N_yz + T)
        s[6] = s[5]
        s[7] = 1. / 4 * m[0, 0, 0] * Pi_yz
        s[8] = s[7]
        s[9] = - 1. / 4 * m[0, 0, 0] * Pi_yz
        s[10] = s[9]
        s[11] = 1. / 4 * m[0, 0, 0] * Pi_xz
        s[12] = s[11]
        s[13] = -1. / 4 * m[0, 0, 0] * Pi_xz
        s[14] = s[13]
        s[15] = 1. / 4 * m[0, 0, 0] * Pi_xy
        s[16] = s[15]
        s[17] = -1. / 4 * m[0, 0, 0] * Pi_xy
        s[18] = s[17]

        return s

    def __call__(self, f):
        # the deletes are not part of the algorithm, they just keep the memory usage lower
        feq = self.lattice.equilibrium(self.lattice.rho(f), self.lattice.u(f))
        # k = torch.zeros_like(f)

        m = self.kbc_moment_transform(f)
        delta_s = self.compute_s_seq_from_m(f, m)

        # k[1] = m[0, 0, 0] / 6. * (3. * m[1, 0, 0])
        # k[0] = m[0, 0, 0]
        # k[2] = -k[1]
        # k[3] = m[0, 0, 0] / 6. * (3. * m[0, 1, 0])
        # k[4] = -k[3]
        # k[5] = m[0, 0, 0] / 6. * (3. * m[0, 0, 1])
        # k[6] = -k[5]

        m = self.kbc_moment_transform(feq)

        delta_s -= self.compute_s_seq_from_m(f, m)
        del m
        delta_h = f - feq - delta_s

        sum_s = self.lattice.rho(delta_s * delta_h / feq)
        sum_h = self.lattice.rho(delta_h * delta_h / feq)
        del feq
        gamma_stab = 1. / self.beta - (2 - 1. / self.beta) * sum_s / sum_h
        gamma_stab[gamma_stab < 1E-15] = 2.0
        # Detect NaN
        gamma_stab[torch.isnan(gamma_stab)] = 2.0
        f = f - self.beta * (2 * delta_s + gamma_stab * delta_h)

        return f
