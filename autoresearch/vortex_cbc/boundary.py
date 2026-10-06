"""
Candidate outflow boundary: delayed interior copy (frozen-flow convective outlet).

Idea: structures leave the domain advected by the mean flow U0 (LU units), i.e.
with velocity U0 cells per step. A ghost column one cell outside the outlet
should therefore carry the state the last interior column had D ~ 1/U0 steps
ago. We keep a ring buffer of the interior macroscopic state (rho, ux, uy),
replay it with delay D at the ghost column, and rebuild the unknown populations
(e_x < 0) from a regularized ansatz: equilibrium of the delayed macro state
plus the non-equilibrium part of the interior. This transports vortical,
entropy and acoustic structures out of the domain with nearly zero reflection
without ever touching the interior fields.
"""
from typing import List

import numpy as np
import torch

import lettuce as lt


class DelayedCopyOutlet(lt.Boundary):
    def __init__(self, flow, delay: int = 17, frac: float = 0.7):
        ctx = flow.context
        e = ctx.convert_to_ndarray(flow.stencil.e)
        self._unknown: List[int] = np.argwhere(e[:, 0] < -0.5).reshape(-1).tolist()
        self._delay = delay
        self._frac = frac
        self._device = ctx.device
        self._history = []

    def __call__(self, flow):
        rho = flow.rho()[0]
        u = flow.u()
        qi = torch.stack([rho[-2], u[0, -2], u[1, -2]])
        self._history.append(qi)
        if len(self._history) > self._delay + 2:
            self._history.pop(0)
        if len(self._history) < self._delay + 2:
            q = qi
        else:
            q = (1.0 - self._frac) * self._history[0] + self._frac * self._history[1]
        rho_g, u_g = rho.clone(), u.clone()
        rho_g[-1], u_g[0, -1], u_g[1, -1] = q[0], q[1], q[2]
        feq = flow.equilibrium(flow, rho_g[..., None], u_g[..., None])[..., 0]
        feq_i = flow.equilibrium(flow, rho[..., None], u[..., None])[..., 0]
        flow.f[:, -1, :] = feq[:, -1, :] + (flow.f[:, -2, :] - feq_i[:, -2, :])
        return flow.f

    def make_no_collision_mask(self, shape, context):
        mask = torch.zeros(shape, dtype=torch.bool, device=self._device)
        mask[-1] = True
        return mask

    def make_no_streaming_mask(self, shape, context):
        mask = torch.zeros(shape, dtype=torch.bool, device=self._device)
        for i in self._unknown:
            mask[i, -1] = True
        return mask

    def native_available(self):
        return False

    def native_generator(self, index):
        raise NotImplementedError


def make_outlet(flow):
    return DelayedCopyOutlet(flow, delay=17)
