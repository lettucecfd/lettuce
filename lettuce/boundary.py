"""
Boundary Conditions.

The `__call__` function of a boundary defines its application to the distribution functions.

Boundary conditions can define a mask (a boolean numpy array)
that specifies the grid points on which the boundary
condition operates.

Boundary classes can define two functions `make_no_stream_mask` and `make_no_collision_mask`
that prevent streaming and collisions on the boundary nodes.

The no-stream mask has the same dimensions as the distribution functions (Q, x, y, (z)) .
The no-collision mask has the same dimensions as the grid (x, y, (z)).

"""

import torch
import numpy as np
from lettuce import (LettuceException)


class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class EquilibriumBoundaryPU:
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """
    def __init__(self, mask, lattice, units, velocity, pressure=0):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)
        self.pressure = lattice.convert_to_tensor(pressure)

    def __call__(self, f):
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask, feq, f)
        return f


class AntiBounceBackOutlet:
    """Allows distributions to leave domain unobstructed through this boundary.
        Based on equations from page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, direction):
        assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
            LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
                                f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        direction = np.array(direction)
        self.lattice = lattice

        #select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)

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
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
            - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
            (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
             - (torch.norm(u_w,dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_stream_mask

    # not 100% sure about this. it probably makes sense to apply collisions here so that f[0] can adapt to the flow
    #def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask
