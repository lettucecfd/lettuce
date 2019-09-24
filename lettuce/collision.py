"""
Collision models
"""

from lettuce.equilibrium import QuadraticEquilibrium
from lettuce.stencils import *
from lettuce.util import LettuceException
from lettuce.lattices import Lattice,LatticeAoS
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


class KBCCollision:
    def __init__(self, lattice, tau):
        self.lattice = lattice
        assert lattice.Q == 27, \
            LettuceException("KBC only realized for D3Q27")
        self.tau = tau
        self.beta = 1. / (2 * tau)

        ##Build a matrix that contains the indices
        self.B = torch.zeros([3, 3, 3, 27], dtype=lattice.dtype)
        if isinstance(lattice,LatticeAoS):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        self.M[i, j, k] = lattice.e[0] ** i * lattice.e[1] ** j * lattice.e[2] ** k
        else:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        self.M[i, j, k] = lattice.e[:,0] ** i * lattice.e[:,1] ** j * lattice.e[:,2] ** k

    def kbc_moment_transform(self,f):
        if isinstance(self.lattice,LatticeAoS):
            m = torch.einsum('abcq,mnoq', self.M, f)
        else:
            m = torch.einsum('abcq,qmno', self.M, f)
        rho = m[0,0,0]
        m /= rho
        m[0,0,0]=rho

        return m

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        m = self.kbc_moment_transform(f)

        T = m[2, 0, 0] + m[0, 2, 0] + m[0, 0, 2]
        N_xz = m[2, 0, 0] - m[0, 0, 2]
        N_yz = m[0, 2, 0] - m[0, 0, 2]
        Pi_xy = m[1, 1, 0]
        Pi_xz = m[1, 0, 1]
        Pi_yz = m[0, 1, 1]

        k = torch.zeros_like(f)
        s = torch.zeros_like(f)
        seq = torch.zeros_like(f)

        k[self.lattice.field(0)] = m[0,0,0]
        k[self.lattice.field(1)] = m[0,0,0] / 6. * (3. * m[1,0,0])
        k[self.lattice.field(2)] = m[0,0,0] / 6. * (3. * -m[1,0,0])
        k[self.lattice.field(3)] = m[0,0,0] / 6. * (3. * m[0,1,0])
        k[self.lattice.field(4)] = m[0,0,0] / 6. * (3. * -m[0,1,0])
        k[self.lattice.field(5)] = m[0,0,0] / 6. * (3. * m[0,0,1])
        k[self.lattice.field(6)] = m[0,0,0] / 6. * (3. * -m[0,0,1])

        s[self.lattice.field(0)] = m[0,0,0] * -T
        s[self.lattice.field(1)] = 1. / 6. * m[0,0,0] * (2 * N_xz - N_yz + T)
        s[self.lattice.field(2)] = 1. / 6. * m[0,0,0] * (2 * N_xz - N_yz + T)
        s[self.lattice.field(3)] = 1. / 6. * m[0,0,0] * (2 * N_yz - N_xz + T)
        s[self.lattice.field(4)] = 1. / 6. * m[0,0,0] * (2 * N_yz - N_xz + T)
        s[self.lattice.field(5)] = 1. / 6. * m[0,0,0] * (-N_xz - N_yz + T)
        s[self.lattice.field(6)] = 1. / 6. * m[0,0,0] * (-N_xz - N_yz + T)
        s[self.lattice.field(7)] = 1. / 4 * m[0,0,0] * Pi_yz
        s[self.lattice.field(8)] = 1. / 4 * m[0,0,0] * Pi_yz
        s[self.lattice.field(9)] = - 1. / 4 * m[0,0,0] * Pi_yz
        s[self.lattice.field(10)] = -1. / 4 * m[0,0,0] * Pi_yz
        s[self.lattice.field(11)] = 1. / 4 * m[0,0,0] * Pi_xz
        s[self.lattice.field(12)] = 1. / 4 * m[0,0,0] * Pi_xz
        s[self.lattice.field(13)] = -1. / 4 * m[0,0,0] * Pi_xz
        s[self.lattice.field(14)] = -1. / 4 * m[0,0,0] * Pi_xz
        s[self.lattice.field(15)] = 1. / 4 * m[0,0,0] * Pi_xy
        s[self.lattice.field(16)] = 1. / 4 * m[0,0,0] * Pi_xy
        s[self.lattice.field(17)] = -1. / 4 * m[0,0,0] * Pi_xy
        s[self.lattice.field(18)] = -1. / 4 * m[0,0,0] * Pi_xy

        m = self.kbc_moment_transform(feq)
        T = m[2, 0, 0] + m[0, 2, 0] + m[0, 0, 2]
        N_xz = m[2, 0, 0] - m[0, 0, 2]
        N_yz = m[0, 2, 0] - m[0, 0, 2]
        Pi_xy = m[1, 1, 0]
        Pi_xz = m[1, 0, 1]
        Pi_yz = m[0, 1, 1]

        seq[self.lattice.field(0)] = m[0,0,0] * -T
        seq[self.lattice.field(1)] = 1. / 6. * m[0,0,0] * (2 * N_xz - N_yz + T)
        seq[self.lattice.field(2)] = 1. / 6. * m[0,0,0] * (2 * N_xz - N_yz + T)
        seq[self.lattice.field(3)] = 1. / 6. * m[0,0,0] * (2 * N_yz - N_xz + T)
        seq[self.lattice.field(4)] = 1. / 6. * m[0,0,0] * (2 * N_yz - N_xz + T)
        seq[self.lattice.field(5)] = 1. / 6. * m[0,0,0] * (-N_xz - N_yz + T)
        seq[self.lattice.field(6)] = 1. / 6. * m[0,0,0] * (-N_xz - N_yz + T)
        seq[self.lattice.field(7)] = 1. / 4 * m[0,0,0] * Pi_yz
        seq[self.lattice.field(8)] = 1. / 4 * m[0,0,0] * Pi_yz
        seq[self.lattice.field(9)] = - 1. / 4 * m[0,0,0] * Pi_yz
        seq[self.lattice.field(10)] = -1. / 4 * m[0,0,0] * Pi_yz
        seq[self.lattice.field(11)] = 1. / 4 * m[0,0,0] * Pi_xz
        seq[self.lattice.field(12)] = 1. / 4 * m[0,0,0] * Pi_xz
        seq[self.lattice.field(13)] = -1. / 4 * m[0,0,0] * Pi_xz
        seq[self.lattice.field(14)] = -1. / 4 * m[0,0,0] * Pi_xz
        seq[self.lattice.field(15)] = 1. / 4 * m[0,0,0] * Pi_xy
        seq[self.lattice.field(16)] = 1. / 4 * m[0,0,0] * Pi_xy
        seq[self.lattice.field(17)] = -1. / 4 * m[0,0,0] * Pi_xy
        seq[self.lattice.field(18)] = -1. / 4 * m[0,0,0] * Pi_xy

        delta_s = s - seq
        delta_h = f - feq - delta_s

        delta_seq = delta_s * delta_h / feq
        sum_s = self.lattice.rho(delta_seq)
        del delta_seq

        delta_heq = delta_h * delta_h / feq
        sum_h = self.lattice.rho(delta_heq)

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
        mnew = m - 1.0/self.tau * (m-meq)
        mnew[0] = m[0] - 1.0/(self.tau+1) * (m[0]-meq[0])
        mnew[self.momentum_indices] = rho*self.u
        f = self.moment_transformation.inverse_transform(mnew)
        return f
