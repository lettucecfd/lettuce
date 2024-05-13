import numpy as np
import torch

from . import AntiBounceBackOutlet

__all__ = ['EquilibriumOutletP']


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """

    def __init__(self, lattice, direction, rho0=1.0):
        AntiBounceBackOutlet.__init__(self, lattice, direction)
        self.rho0 = rho0

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = self.rho0 * torch.ones_like(rho[here])
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[..., None], u_w[..., None])[..., 0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, f_shape):
        no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask

    def native_available(self) -> bool:
        return False
