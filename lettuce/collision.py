"""
Collision models
"""

import torch

from lettuce.equilibrium import QuadraticEquilibrium
from lettuce.util import LettuceException


class BGKCollision:
    def __init__(self, lattice, tau):
        self.lattice = lattice
        self.tau = tau

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        f = f - 1.0 / self.tau * (f - feq)
        return f


class MRTCollision:
    """Multiple relaxation time collision operator

    This is an MRT operator in the most general sense of the word.
    The transform does not have to be linear and can, e.g., be any moment or cumulant transform.
    """

    def __init__(self, lattice, transform, relaxation_parameters):
        self.lattice = lattice
        self.transform = transform
        self.relaxation_parameters = lattice.convert_to_tensor(relaxation_parameters)

    def __call__(self, f):
        m = self.transform.transform(f)
        meq = self.transform.equilibrium(m)
        m = m - self.lattice.einsum("q,q->q", [1 / self.relaxation_parameters, m - meq])
        f = self.transform.inverse_transform(m)
        return f


class TRTCollision:
    """Two relaxation time collision model - standard implementation (cf. Krüger 2017)
        """

    def __init__(self, lattice, tau, tau_minus=1.0):
        self.lattice = lattice
        self.tau_plus = tau
        self.tau_minus = tau_minus

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        f_diff_neq = ((f + f[self.lattice.stencil.opposite]) - (feq + feq[self.lattice.stencil.opposite])) / (
                2.0 * self.tau_plus)
        f_diff_neq += ((f - f[self.lattice.stencil.opposite]) - (feq - feq[self.lattice.stencil.opposite])) / (
                2.0 * self.tau_minus)
        f = f - f_diff_neq
        return f

class RegularizedCollision:
    def __init__(self, lattice, tau):
        self.lattice = lattice
        self.tau = tau
        self.Q_matrix = torch.zeros([lattice.Q, lattice.D, lattice.D], device=lattice.device, dtype=lattice.dtype)

        for a in range(lattice.Q):
            for b in range(lattice.D):
                for c in range(lattice.D):
                    self.Q_matrix[a, b, c] = lattice.e[a, b] * lattice.e[a, c]
                    if b==c:
                        self.Q_matrix[a, b, c] -= lattice.cs * lattice.cs

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        pi_neq = self.lattice.shear_tensor(f) - self.lattice.shear_tensor(feq)
        cs4 = self.lattice.cs * self.lattice.cs * self.lattice.cs * self.lattice.cs

        pi_neq = self.lattice.einsum("qab,ab->q", [self.Q_matrix, pi_neq])
        pi_neq = self.lattice.einsum("q,q->q", [self.lattice.w,pi_neq])

        fi1 = pi_neq / (2 * cs4)
        f = feq + (1. - 1. / self.tau) * fi1

        return f


class KBCCollision:
    """Entropic multi-relaxation time-relaxation time model according to Karlin et al."""
    def __init__(self, lattice, tau):
        self.lattice = lattice
        assert lattice.Q == 27, \
            LettuceException("KBC only realized for D3Q27")
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
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        k = torch.zeros_like(f)

        m = self.kbc_moment_transform(f)
        delta_s = self.compute_s_seq_from_m(f, m)

        k[1] = m[0, 0, 0] / 6. * (3. * m[1, 0, 0])
        k[0] = m[0, 0, 0]
        k[2] = -k[1]
        k[3] = m[0, 0, 0] / 6. * (3. * m[0, 1, 0])
        k[4] = -k[3]
        k[5] = m[0, 0, 0] / 6. * (3. * m[0, 0, 1])
        k[6] = -k[5]

        m = self.kbc_moment_transform(feq)

        delta_s -= self.compute_s_seq_from_m(f, m)
        delta_h = f - feq - delta_s

        sum_s = self.lattice.rho(delta_s * delta_h / feq)
        sum_h = self.lattice.rho(delta_h * delta_h / feq)

        gamma_stab = 1. / self.beta - (2 - 1. / self.beta) * sum_s / sum_h
        f = f - self.beta * (2 * delta_s + gamma_stab * delta_h)

        return f


class BGKInitialization:
    """Keep velocity constant."""

    def __init__(self, lattice, flow, moment_transformation):
        self.lattice = lattice
        self.tau = flow.units.relaxation_parameter_lu
        self.moment_transformation = moment_transformation
        p, u = flow.initial_solution(flow.grid)
        self.u = flow.units.convert_velocity_to_lu(lattice.convert_to_tensor(u))
        self.rho0 = flow.units.characteristic_density_lu
        self.equilibrium = QuadraticEquilibrium(self.lattice)
        momentum_names = tuple([f"j{x}" for x in "xyz"[:self.lattice.D]])
        self.momentum_indices = moment_transformation[momentum_names]

    def __call__(self, f):
        rho = self.lattice.rho(f)
        feq = self.equilibrium(rho, self.u)
        m = self.moment_transformation.transform(f)
        meq = self.moment_transformation.transform(feq)
        mnew = m - 1.0 / self.tau * (m - meq)
        mnew[0] = m[0] - 1.0 / (self.tau + 1) * (m[0] - meq[0])
        mnew[self.momentum_indices] = rho * self.u
        f = self.moment_transformation.inverse_transform(mnew)
        return f
