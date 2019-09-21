"""
Collision models
"""

from lettuce.equilibrium import QuadraticEquilibrium
from lettuce.stencils import *
import torch


class BGKCollision:
    def __init__(self, lattice, tau):
        self.lattice = lattice
        self.tau = tau

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        f = f - 1.0/self.tau * (f-feq)
        return f


class KBCCollision:
    def __init__(self, lattice, tau):
        self.lattice = lattice
        assert lattice.Q == 27, \
            LettuceException("KBC only realized for D3Q27")
        self.tau = tau
        self.beta = 1. / (2 * tau)

        ##Build a matrix that contains the indices
        B = np.zeros([3, 3, 3, 27])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    B[i, j, k] = lattice.e[:, 0] ** i * lattice.e[:, 1] ** j * lattice.e[:, 2] ** k
        self.M = self.lattice.convert_to_tensor(B)

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        m = torch.einsum('ijkl,lmno', [self.M, f]) / rho

        T = m[2, 0, 0] + m[0, 2, 0] + m[0, 0, 2]
        N_xz = m[2, 0, 0] - m[0, 0, 2]
        N_yz = m[0, 2, 0] - m[0, 0, 2]
        Pi_xy = m[1, 1, 0]
        Pi_xz = m[1, 0, 1]
        Pi_yz = m[0, 1, 1]

        k = torch.zeros_like(f)
        s = torch.zeros_like(f)
        seq = torch.zeros_like(f)

        k[0] = rho
        k[1] = rho / 6. * (3. * u[0])
        k[2] = rho / 6. * (3. * -u[0])
        k[3] = rho / 6. * (3. * u[1])
        k[4] = rho / 6. * (3. * -u[1])
        k[5] = rho / 6. * (3. * u[2])
        k[6] = rho / 6. * (3. * -u[2])

        s[0] = rho * -T
        s[1] = 1. / 6. * rho * (2 * N_xz - N_yz + T)
        s[2] = 1. / 6. * rho * (2 * N_xz - N_yz + T)
        s[3] = 1. / 6. * rho * (2 * N_yz - N_xz + T)
        s[4] = s[3]
        s[5] = 1. / 6. * rho * (-N_xz - N_yz + T)
        s[6] = s[5]
        s[7] = 1. / 4 * rho * Pi_yz
        s[8] = 1. / 4 * rho * Pi_yz
        s[9] = - 1. / 4 * rho * Pi_yz
        s[10] = -1. / 4 * rho * Pi_yz
        s[11] = 1. / 4 * rho * Pi_xz
        s[12] = 1. / 4 * rho * Pi_xz
        s[13] = -1. / 4 * rho * Pi_xz
        s[14] = -1. / 4 * rho * Pi_xz
        s[15] = 1. / 4 * rho * Pi_xy
        s[16] = 1. / 4 * rho * Pi_xy
        s[17] = -1. / 4 * rho * Pi_xy
        s[18] = -1. / 4 * rho * Pi_xy

        h = f - k - s

        m = torch.einsum('ijkl,lmno', self.M, feq) / rho
        T = m[2, 0, 0] + m[0, 2, 0] + m[0, 0, 2]
        N_xz = m[2, 0, 0] - m[0, 0, 2]
        N_yz = m[0, 2, 0] - m[0, 0, 2]
        Pi_xy = m[1, 1, 0]
        Pi_xz = m[1, 0, 1]
        Pi_yz = m[0, 1, 1]

        seq[0] = rho * -T
        seq[1] = 1. / 6. * rho * (2 * N_xz - N_yz + T)
        seq[2] = 1. / 6. * rho * (2 * N_xz - N_yz + T)
        seq[3] = 1. / 6. * rho * (2 * N_yz - N_xz + T)
        seq[4] = seq[3]
        seq[5] = 1. / 6. * rho * (-N_xz - N_yz + T)
        seq[6] = seq[5]
        seq[7] = 1. / 4 * rho * Pi_yz
        seq[8] = 1. / 4 * rho * Pi_yz
        seq[9] = - 1. / 4 * rho * Pi_yz
        seq[10] = -1. / 4 * rho * Pi_yz
        seq[11] = 1. / 4 * rho * Pi_xz
        seq[12] = 1. / 4 * rho * Pi_xz
        seq[13] = -1. / 4 * rho * Pi_xz
        seq[14] = -1. / 4 * rho * Pi_xz
        seq[15] = 1. / 4 * rho * Pi_xy
        seq[16] = 1. / 4 * rho * Pi_xy
        seq[17] = -1. / 4 * rho * Pi_xy
        seq[18] = -1. / 4 * rho * Pi_xy

        heq = feq - k - seq

        delta_s = s - seq
        delta_h = h - heq

        delta_seq = delta_s * delta_h / feq
        delta_heq = delta_h * delta_h / feq

        sum_s = delta_seq.sum(0)
        sum_h = delta_heq.sum(0)

        gamma_stab = 1. / self.beta - (2 - 1. / self.beta) * sum_s / sum_h

        f = f - self.beta * (2 * delta_s + gamma_stab * delta_h)

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
        m = m - self.lattice.einsum("q,q->q", [1/self.relaxation_parameters, m-meq])
        f = self.transform.inverse_transform(m)
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
        mnew = m - 1.0/self.tau * (m-meq)
        mnew[0] = m[0] - 1.0/(self.tau+1) * (m[0]-meq[0])
        mnew[self.momentum_indices] = rho*self.u
        f = self.moment_transformation.inverse_transform(mnew)
        return f
