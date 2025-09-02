import warnings

import torch

from ... import Flow, Collision, TorchStencil
from .. import D3Q27, D2Q9

__all__ = ['KBCCollision3D', 'KBCCollision2D', 'KBCCollision']


class KBCCollision(Collision):
    """
    Entropic multi-relaxation time-relaxation time model according to Karlin
    et al.
    """

    def __init__(self, tau: float = None):
        self.tau = tau
        self.beta = None
        self.M = None

    def kbc_moment_transform(self, f):
        pass

    def kbc_moment_transform_3d(self, f):
        """Transforms the f into the KBC moment representation"""
        m = torch.einsum('abcq,qmno', self.M, f)
        rho = m[0, 0, 0]
        m = m / rho
        m[0, 0, 0] = rho
        return m

    def kbc_moment_transform_2d(self, f):
        """Transforms the f into the KBC moment representation"""
        m = torch.einsum('abq,qmn', self.M, f)
        rho = m[0, 0]
        m = m / rho
        m[0, 0] = rho
        return m

    def compute_s_seq_from_m(self, f, m):
        pass

    def compute_s_seq_from_m_3d(self, f, m):
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

    def compute_s_seq_from_m_2d(self, f, m):
        s = torch.zeros_like(f)

        T = m[2, 0] + m[0, 2]
        N = m[2, 0] - m[0, 2]

        Pi_xy = m[1, 1]

        s[0] = m[0, 0] * -T
        s[1] = 1. / 2. * m[0, 0] * (0.5 * (T + N))
        s[2] = 1. / 2. * m[0, 0] * (0.5 * (T - N))
        s[3] = 1. / 2. * m[0, 0] * (0.5 * (T + N))
        s[4] = 1. / 2. * m[0, 0] * (0.5 * (T - N))
        s[5] = 1. / 4. * m[0, 0] * Pi_xy
        s[6] = -s[5]
        s[7] = 1. / 4 * m[0, 0] * Pi_xy
        s[8] = -s[7]

        return s

    def __call__(self, flow: 'Flow') -> torch.Tensor:
        if self.M is None:
            self.tau = flow.units.relaxation_parameter_lu
            self.beta = 1. / (2 * self.tau)
            if flow.stencil.d == 3:
                assert isinstance(flow.stencil, D3Q27), \
                    "KBC Collision is only implemented for D3Q27!"
                self.compute_s_seq_from_m = self.compute_s_seq_from_m_3d
                self.kbc_moment_transform = self.kbc_moment_transform_3d
                # Build a matrix that contains the indices
                self.M = flow.context.zero_tensor([3, 3, 3, 27])
                torch_stencil = TorchStencil(D3Q27(), flow.context)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            self.M[i, j, k] = (torch_stencil.e[:, 0] ** i *
                                               torch_stencil.e[:, 1] ** j *
                                               torch_stencil.e[:, 2] ** k)
            elif flow.stencil.d == 2:
                assert isinstance(flow.stencil, D2Q9), \
                    "KBC Collision is only implemented for D2Q9!"
                self.compute_s_seq_from_m = self.compute_s_seq_from_m_2d
                self.kbc_moment_transform = self.kbc_moment_transform_2d
                # Build a matrix that contains the indices
                self.M = flow.context.zero_tensor([3, 3, 9])
                torch_stencil = TorchStencil(D2Q9(), flow.context)
                for i in range(3):
                    for j in range(3):
                        self.M[i, j] = (torch_stencil.e[:, 0] ** i
                                        * torch_stencil.e[:, 1] ** j)
            else:
                raise NotImplementedError("KBC Collision is only implemented "
                                          "for 2d and 3d!")
        # the deletes are not part of the algorithm, they just keep the memory
        # usage lower
        feq = flow.equilibrium(flow)
        # k = torch.zeros_like(f)

        m = self.kbc_moment_transform(flow.f)
        delta_s = self.compute_s_seq_from_m(flow.f, m)

        # k[1] = m[0, 0, 0] / 6. * (3. * m[1, 0, 0])
        # k[0] = m[0, 0, 0]
        # k[2] = -k[1]
        # k[3] = m[0, 0, 0] / 6. * (3. * m[0, 1, 0])
        # k[4] = -k[3]
        # k[5] = m[0, 0, 0] / 6. * (3. * m[0, 0, 1])
        # k[6] = -k[5]

        m = self.kbc_moment_transform(feq)
        delta_s -= self.compute_s_seq_from_m(flow.f, m)
        del m

        delta_h = flow.f - feq - delta_s
        sum_s = flow.rho(delta_s * delta_h / feq)
        sum_h = flow.rho(delta_h * delta_h / feq)
        del feq

        gamma_stab = 1. / self.beta - (2 - 1. / self.beta) * sum_s / sum_h
        gamma_stab[gamma_stab < 1E-15] = 2.0
        # Detect NaN
        gamma_stab[torch.isnan(gamma_stab)] = 2.0
        f = flow.f - self.beta * (2 * delta_s + gamma_stab * delta_h)

        return f

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int) -> 'NativeCollision':
        pass


class KBCCollision2D(KBCCollision):
    def __init__(self, tau: float = None):
        warnings.warn("KBCCollision2D is is deprecated! Use KBCCollision "
                      "instead!")
        super().__init__()


class KBCCollision3D(KBCCollision):
    def __init__(self, tau: float = None):
        warnings.warn("KBCCollision2D is is deprecated! Use KBCCollision "
                      "instead!")
        super().__init__()
