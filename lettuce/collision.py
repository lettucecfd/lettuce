"""
Collision models
"""

import torch
import numpy as np

from lettuce.equilibrium import QuadraticEquilibrium
from lettuce.moments import DEFAULT_TRANSFORM
from lettuce.util import LettuceCollisionNotDefined, LettuceInvalidNetworkOutput
from lettuce.stencils import D2Q9, D3Q27
from lettuce.symmetry import SymmetryGroup

__all__ = [
    "Collision",
    "BGKCollision", "KBCCollision2D", "KBCCollision3D", "MRTCollision",
    "RegularizedCollision", "SmagorinskyCollision", "TRTCollision", "BGKInitialization",
    "EquivariantNeuralCollision"
]


class Collision:
    def __call__(self, f):
        return NotImplemented


class BGKCollision(Collision):
    def __init__(self, lattice, tau, force=None):
        self.force = force
        self.lattice = lattice
        self.tau = tau

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u_eq = 0 if self.force is None else self.force.u_eq(f)
        u = self.lattice.u(f) + u_eq
        feq = self.lattice.equilibrium(rho, u)
        Si = 0 if self.force is None else self.force.source_term(u)
        return f - 1.0 / self.tau * (f - feq) + Si


class MRTCollision(Collision):
    """Multiple relaxation time collision operator

    This is an MRT operator in the most general sense of the word.
    The transform does not have to be linear and can, e.g., be any moment or cumulant transform.
    """

    def __init__(self, lattice, relaxation_parameters, transform=None):
        self.lattice = lattice
        if transform is None:
            try:
                self.transform = DEFAULT_TRANSFORM[lattice.stencil](lattice)
            except KeyError:
                raise LettuceCollisionNotDefined("No entry for stencil {lattice.stencil} in moments.DEFAULT_TRANSFORM")
        else:
            self.transform = transform
        if isinstance(relaxation_parameters, float):
            tau = relaxation_parameters
            self.relaxation_parameters = lattice.convert_to_tensor(tau * np.ones(lattice.stencil.Q()))
        else:
            self.relaxation_parameters = lattice.convert_to_tensor(relaxation_parameters)

    def __call__(self, f):
        m = self.transform.transform(f)
        meq = self.transform.equilibrium(m)
        m = m - self.lattice.einsum("q,q->q", [1 / self.relaxation_parameters, m - meq])
        f = self.transform.inverse_transform(m)
        return f


class TRTCollision(Collision):
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


class RegularizedCollision(Collision):
    """Regularized LBM according to Jonas Latt and Bastien Chopard (2006)"""

    def __init__(self, lattice, tau):
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
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        pi_neq = self.lattice.shear_tensor(f - feq)
        cs4 = self.lattice.cs ** 4

        pi_neq = self.lattice.einsum("qab,ab->q", [self.Q_matrix, pi_neq])
        pi_neq = self.lattice.einsum("q,q->q", [self.lattice.w, pi_neq])

        fi1 = pi_neq / (2 * cs4)
        f = feq + (1. - 1. / self.tau) * fi1

        return f


class KBCCollision2D(Collision):
    """Entropic multi-relaxation time model according to Karlin et al. in two dimensions"""

    def __init__(self, lattice, tau):
        self.lattice = lattice
        if not lattice.stencil == D2Q9:
            raise LettuceCollisionNotDefined("This implementation only works for the D2Q9 stencil.")
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
        feq = self.lattice.equilibrium(self.lattice.rho(f), self.lattice.u(f))
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


class KBCCollision3D(Collision):
    """Entropic multi-relaxation time-relaxation time model according to Karlin et al. in three dimensions"""

    def __init__(self, lattice, tau):
        self.lattice = lattice
        if not lattice.stencil == D3Q27:
            raise LettuceCollisionNotDefined("This implementation only works for the D3Q27 stencil.")
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


class SmagorinskyCollision(Collision):
    """Smagorinsky large eddy simulation (LES) collision model with BGK operator."""

    def __init__(self, lattice, tau, smagorinsky_constant=0.17, force=None):
        self.force = force
        self.lattice = lattice
        self.tau = tau
        self.iterations = 2
        self.tau_eff = tau
        self.constant = smagorinsky_constant

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u_eq = 0 if self.force is None else self.force.u_eq(f)
        u = self.lattice.u(f) + u_eq
        feq = self.lattice.equilibrium(rho, u)
        S_shear = self.lattice.shear_tensor(f - feq)
        S_shear /= (2.0 * rho * self.lattice.cs ** 2)
        self.tau_eff = self.tau
        nu = (self.tau - 0.5) / 3.0

        for i in range(self.iterations):
            S = S_shear / self.tau_eff
            S = self.lattice.einsum('ab,ab->', [S, S])
            nu_t = self.constant ** 2 * S
            nu_eff = nu + nu_t
            self.tau_eff = nu_eff * 3.0 + 0.5
        Si = 0 if self.force is None else self.force.source_term(u)
        return f - 1.0 / self.tau_eff * (f - feq) + Si


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


class EquivariantNeuralCollision(torch.nn.Module):
    """An MRT model that is equivariant under the lattice symmetries by relaxing all moments of the same
    order with the same rate.

    Parameters
    ----------
    default_tau : float
        The default relaxation parameter operating on all moment orders for which the tau_net
        does not produce output. See documentation there.
    tau_net : torch.nn.Module
        A network that receives moments and returns unconstrained relaxation parameters for the highest-order moments.
        The input shape to the network is (..., Q), where "..." is any number of batch and grid dimensions
        and Q is the number of discrete distributions at each node.
        The output shape is (..., N), where N is the number of moment ORDERS, whose relaxation is prescribed
        by the network. Only the N highest moment orders will be relaxed.
        Note that the output of the network should be unconstrained and will be rendered > 0.5 by this class.
    moment_transform : Transform
        The moment transformation.

    """
    def __init__(self, default_tau, tau_net, moment_transform):
        super().__init__()
        self.trafo = moment_transform
        self.lattice = moment_transform.lattice
        self.tau = default_tau
        self.net = tau_net.to(dtype=self.lattice.dtype, device=self.lattice.device)
        # symmetries
        symmetry_group = SymmetryGroup(moment_transform.lattice.stencil)
        self.rep = symmetry_group.moment_representations(moment_transform)
        # infer moment order from moment name
        self.moment_order = np.array([sum(name.count(x) for x in "xyz") for name in moment_transform.names])
        self.last_taus = None

    @staticmethod
    def gt_half(a):
        """transform into a value > 0.5"""
        return 0.5 + torch.exp(a)

    def _compute_relaxation_parameters(self, m):
        # default taus
        taus = self.tau * torch.ones_like(m)
        # compute m under all symmetry group representations
        y = torch.einsum(
            f"npq, ...q{'xyz'[:self.lattice.D]} -> n...{'xyz'[:self.lattice.D]}p",
            self.rep, m
        )
        # compute higher-order taus from neural network
        y = self.net(y).sum(0)
        # move Q-axis in front of grid axes
        q_dim = len(y.shape) - 1 - self.lattice.D
        tau = y.moveaxis(len(y.shape) - 1, q_dim)
        # render tau > 0.5
        tau = self.gt_half(tau)
        # apply learned taus to highest order moments
        moment_orders = np.sort(np.unique(self.moment_order))
        if not len(moment_orders) >= tau.shape[q_dim]:
            raise LettuceInvalidNetworkOutput(
                f"Network produced {tau.shape[q_dim]} taus but only {len(moment_orders)} "
                f"are available. Moments of each order are relaxed with the same tau."
            )
        learned_tau_moment_orders = moment_orders[-tau.shape[q_dim]:]
        for i, order in enumerate(learned_tau_moment_orders):
            taus[self.moment_order == order] = tau[i]
        return taus

    def forward(self, f):
        m = self.trafo.transform(f)
        taus = self._compute_relaxation_parameters(m)
        self.last_taus = taus
        meq = self.trafo.equilibrium(m)
        m_postcollision = m - 1. / taus * (m - meq)
        return self.trafo.inverse_transform(m_postcollision)


