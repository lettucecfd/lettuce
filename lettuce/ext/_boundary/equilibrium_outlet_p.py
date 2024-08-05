from typing import List, Optional

import numpy as np
import torch

__all__ = ['EquilibriumOutletP']

from ... import Flow, Context, Boundary


class EquilibriumOutletP(Boundary):
    """Equilibrium outlet with constant pressure.
    """

    def __init__(self, flow: 'Flow', context: 'Context', direction: List[int],
                 rho_outlet: float = 1.0):
        self.rho_outlet = context.convert_to_tensor(rho_outlet)
        self.context = context

        assert len(direction) in [1, 2, 3], \
            (f"Invalid direction parameter. Expected direction of of length 1,"
             f" 2 or 3 but got {len(direction)}.")

        assert ((direction.count(0) == (len(direction) - 1))
                and ((1 in direction) ^ (-1 in direction))), \
            (f"Invalid direction parameter. Expected direction with all "
             f"entries 0 except one 1 or -1 but got {direction}.")

        direction = np.array(direction)

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(flow.stencil.e, direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        # construct indices for einsum and get w in proper shape for the calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = flow.torch_stencil.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = flow.torch_stencil.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = flow.torch_stencil.w[self.velocities]

    def __call__(self, flow: 'Flow'):
        outlet = [slice(None)] + self.index
        neighbor = [slice(None)] + self.neighbor
        rho_outlet = self.rho_outlet * torch.ones_like(flow.rho()[outlet])
        feq = flow.equilibrium(flow, rho_outlet[..., None],
                               flow.u()[neighbor][..., None])
        return flow.einsum("q,q->q", [feq,
                                      torch.ones_like(flow.f)])

    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        no_streaming_mask = context.zero_tensor(shape, dtype=bool)
        no_streaming_mask[[np.setdiff1d(np.arange(shape[0]), self.velocities)]
                          + self.index] = 1
        return no_streaming_mask

    def make_no_collision_mask(self, shape: List[int], context: 'Context'):
        no_collision_mask = context.zero_tensor(shape, dtype=torch.bool)
        no_collision_mask[self.index] = 1
        return no_collision_mask

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int):
        pass
