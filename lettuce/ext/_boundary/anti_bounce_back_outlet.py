from typing import List

import numpy as np
import torch

from ... import Boundary
from ... import TorchStencil

__all__ = ['AntiBounceBackOutlet']


class AntiBounceBackOutlet(Boundary):
    """Allows distributions to leave domain unobstructed through this boundary.
    Based on equations from page 195 of "The lattice Boltzmann method"
    (2016 by Krüger et al.) Give the side of the domain with the boundary as
    list [x, y, z] with only one entry nonzero
    [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
    [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
    """

    def __init__(self, direction: [List[int]], stencil: 'TorchStencil'):
        assert len(direction) in [1, 2, 3], \
            (f"Invalid direction parameter. Expected direction of of length "
             f"1, 2 or 3 but got {len(direction)}.")

        assert ((direction.tolist().count(0) == (len(direction) - 1))
                and ((1 in direction) ^ (-1 in direction))), \
            (f"Invalid direction parameter. Expected direction with all "
             f"entries 0 except one 1 or -1 but got {direction}.")

        self.stencil = stencil

        # select velocities to be bounced (the ones pointing in "direction")
        # self.velocities = torch.argwhere(
        #     torch.eq(torch.matmul(stencil.e, direction), 1))[None, :]
        self.velocities = np.concatenate(np.argwhere(
            np.matmul(self.stencil.e.cpu().numpy(), direction.cpu().numpy()) >
            1 - 1e-6),
            axis=0)

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
        # construct indices for einsum and get w in proper shape for the
        # calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = stencil.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = stencil.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = stencil.w[self.velocities]

    def __call__(self, flow: 'Flow'):
        # TODO: not 100% sure about this. But collision seem to stabilize the
        #  boundary.
        # f = self.collision(flow.f)
        u = flow.u()
        u_w = (u[[slice(None)] + self.index]
               + 0.5 * (u[[slice(None)] + self.index]
                        - u[[slice(None)] + self.neighbor]))
        f = flow.f
        flow.f[[np.array(flow.stencil.opposite)[self.velocities]]
               + self.index] = (
                - flow.f[[self.velocities] + self.index]
                + self.w
                * flow.rho()[[slice(None)] + self.index]
                * (2
                   + torch.einsum(self.dims,
                                  flow.torch_stencil.e[self.velocities],
                                  u_w) ** 2 / flow.stencil.cs ** 4
                   - (torch.norm(u_w, dim=0) / flow.stencil.cs) ** 2)
        )
        return f

    def make_no_streaming_mask(self, f_shape, context: 'Context'):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool,
                                     device=context.device)
        no_stream_mask[[self.stencil.opposite.cpu().numpy()[
                            self.velocities]] +
                       self.index] = 1
        # no_stream_mask[[self.stencil.opposite[self.velocities].to(
        #     dtype=torch.uint8)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, shape: List[int], context: 'Context'):
        no_collision_mask = context.zero_tensor(shape, dtype=bool)
        no_collision_mask[self.index] = 1
        return no_collision_mask

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int) -> 'NativeBoundary':
        pass
