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

__all__ = ["BounceBackBoundary", "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumOutletP",
           "FlippedBoundary", "TGV3D", "superTGV3D", "newsuperTGV3D"]


class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)  # "Punkte an denen self.mask, also randpunkte liegen, werden mit f[] bezogen und andere mit f"
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

        assert isinstance(direction, list), \
            LettuceException(
                f"Invalid direction parameter. Expected direction of type list but got {type(direction)}.")

        assert len(direction) in [1, 2, 3], \
            LettuceException(
                f"Invalid direction parameter. Expected direction of of length 1, 2 or 3 but got {len(direction)}.")

        assert (direction.count(0) == (len(direction) - 1)) and ((1 in direction) ^ (-1 in direction)), \
            LettuceException(
                "Invalid direction parameter. "
                f"Expected direction with all entries 0 except one 1 or -1 but got {direction}.")

        direction = np.array(direction)
        self.lattice = lattice

        # select velocities to be bounced (the ones pointing in "direction")
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
                 - (torch.norm(u_w, dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_stream_mask

    # not 100% sure about this. But collisions seem to stabilize the boundary.
    # def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """

    def __init__(self, lattice, direction, rho0=1.0):
        super(EquilibriumOutletP, self).__init__(lattice, direction)
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

# class SlipBoundary:
#     def __init__(self, mask, lattice):
#         self.mask = lattice.convert_to_tensor(mask)
#         self.lattice = lattice
#
#     def __call__(self, f):
#         self.max_col = self.mask.shape[1]
#         self.max_row = self.mask.shape[0]
#         self.resolution=self.max_col
#         self.opposite_x = [0, 1, 4, 3, 2, 8, 7, 6, 5]
#         self.opposite_y = [0, 3, 2, 1, 4, 6, 5, 8, 7]
#         self.opposite_xy = [0, 3, 4, 1, 2, 7, 8, 5, 6]
#
#         self.row_indices = torch.arange(self.max_row).unsqueeze(0)#unsqueeze nachschlagen
#         self.col_indices = torch.arange(self.max_col).unsqueeze(1)
#         "F-Bedingungen um 1 nach oben oder unten oder garnicht verschieben, damit die verschiebung der f-Werte einfacher wird"
#
#
#         return f
class FlippedBoundary:
    #def __init__(self, lattice):
        #self.mask = lattice.convert_to_tensor(mask)
        #self.lattice = lattice

    def __call__(self, f):

        # self.saver=f[7,:,0].clone()
        # f[7,:,0]=f[6,:,-1]
        # f[6,:,-1]=self.saver
        # self.saver=f[4,:,0].clone()
        # f[4,:,0]=f[2,:,-1]
        # f[2,:,-1]=self.saver
        # self.saver=f[8,:,0].clone()
        # f[8,:,0]=f[5,:,-1]
        # f[5,:,-1]=self.saver
        #
        # self.saver=f[6,0,:].clone()
        # f[6,0,:]=f[5,-1,:]
        # f[5,-1,:]=self.saver
        # self.saver=f[3,0,:].clone()
        # f[3,0,:]=f[1,-1,:]
        # f[1,-1,:]=self.saver
        # self.saver=f[7,0,:].clone()
        # f[7,0,:]=f[8,-1,:]
        # f[8,-1,:]=self.saver
        return f
class TGV3D:

    def __init__(self, lattice):
     self.lattice = lattice
    def __call__(self, f):



       for row in range(len(self.lattice.stencil.switch_xz)):
          self.saver=f[self.lattice.stencil.switch_xz[row][0],:,0,:].clone()
          f[self.lattice.stencil.switch_xz[row][0],:,0,:]=f[self.lattice.stencil.switch_xz[row][1],:,-1,:].clone()
          f[self.lattice.stencil.switch_xz[row][1],:,-1,:] = self.saver


       for row in range(len(self.lattice.stencil.switch_yz)):
          self.saver = f[self.lattice.stencil.switch_yz[row][0], 0, :, :].clone()
          f[self.lattice.stencil.switch_yz[row][0], 0, :, :] = f[self.lattice.stencil.switch_yz[row][1], -1, :, :].clone()
          f[self.lattice.stencil.switch_yz[row][1], -1, :, :] = self.saver




       for row in range(len(self.lattice.stencil.switch_x)):
           self.saver = torch.flip(f[self.lattice.stencil.switch_x[row][0],:,:,0].clone(),dims=[1])
           f[self.lattice.stencil.switch_x[row][0],:,:,0]=torch.flip(f[self.lattice.stencil.switch_x[row][1],:,:,-1].clone(),dims=[1])
           f[self.lattice.stencil.switch_x[row][1],:,:, -1]=self.saver



       # f[5,:,:,-1]=torch.flip(f[5,:,:,-1].clone(),[1])
       # f[11, :, :, -1] = torch.flip(f[11, :, :, -1].clone(), [1])
       # f[14, :, :, -1] = torch.flip(f[14, :, :, -1].clone(), [1])
       # self.saver=f[7,:,:,-1].clone()
       # f[7,:,:,-1]=f[10,:,:,-1]
       # f[10,:,:,-1]=self.saver
       # f[6,:,:,0]=torch.flip(f[6,:,:,0].clone(),[1])
       # f[12, :, :, 0] = torch.flip(f[12, :, :, 0].clone(), [1])
       # f[13, :, :, 0] = torch.flip(f[13, :, :, 0].clone(), [1])
       # self.saver=f[8,:,:,0].clone()
       # f[8,:,:,0]=f[9,:,:,0]
       # f[9,:,:,0]=self.saver

       return f

class superTGV3D:

    def __init__(self, lattice):
     self.lattice = lattice
    def __call__(self, f):
      self.fclone=f.clone()


      for row in range(len(self.lattice.stencil.switch_rotyx)):
          f[self.lattice.stencil.switch_rotyx[row][1],:,0,:]= self.fclone[self.lattice.stencil.switch_rotyx[row][0],-1,:,:]
      for row in range(len(self.lattice.stencil.switch_rotxy)):
          f[self.lattice.stencil.switch_rotxy[row][1],0,:,:]= self.fclone[self.lattice.stencil.switch_rotxy[row][0],:,-1,:]

      for row in range(len(self.lattice.stencil.switch_xz)):
          f[self.lattice.stencil.switch_xz[row][1],:,-1,:]=self.fclone[self.lattice.stencil.switch_xz[row][0],:,0,:]
      for row in range(len(self.lattice.stencil.switch_yz)):
          f[self.lattice.stencil.switch_yz[row][1], -1, :, :] = self.fclone[self.lattice.stencil.switch_yz[row][0], 0, :, :]


      for row in range(len(self.lattice.stencil.switch_xy)):
          f[self.lattice.stencil.switch_xy[row][0], :,:,0] = self.fclone[self.lattice.stencil.switch_xy[row][1], :,:,-1]

      for row in range(len(self.lattice.stencil.switch_diagonal2)):

          f[self.lattice.stencil.switch_diagonal2[row][1],:,:,-1]=torch.transpose(self.fclone[self.lattice.stencil.switch_diagonal2[row][0],:,:,0], 0, 1)

      f[18,0,-1,:]=self.fclone[18,0,-1,:]
      f[17,-1,0,:]=self.fclone[17,-1,0,:]
      f[16,0,0,:]=self.fclone[15,-1,-1,:]
      f[15,-1,-1,:]=self.fclone[16,0,0,:]
      #for row in range((len(self.lattice.stencil.switch_xy)))
      f[9,:,-1,0]=self.fclone[10,:,0,-1]
      f[13,-1,:,0]=self.fclone[14,0,:,-1]
      f[12,0,:,0]=self.fclone[7,:,-1,-1]
      f[8,:,0,0]=self.fclone[11,-1,:,-1]

      f[10,:,0,-1]=self.fclone[9,:,-1,0]
      f[7,:,-1,-1]=self.fclone[12,0,:,0]

      f[11,-1,:,-1]=self.fclone[8,:,0,0]
      f[14,0,:,-1]=self.fclone[13,-1,:,0]
      #for
      return f

class newsuperTGV3D:
    def __init__(self, lattice):
     self.lattice = lattice

    def __call__(self, f):

        self.e=self.lattice.stencil.e
        #self.sym_search=self.lattice

        #self.switch_stencil_wall=self.sym_search.switch_stencil_wall
        #self.switch_stencil_borders = self.sym_search.switch_stencil_borders
        #self.switch_stencil_corner = self.sym_search.switch_stencil_array

        if 'switch_stencil_wall' not in locals():
        #####################################
            self.s_a = np.array([[0, -1, -1, 0, 1, 1, 1, 2],
                                 [1, -1, 1, 0, -1, 1, 1, 2],
                                 [2, 1, 1, 0, 1, 1, -1, 2],
                                 [2, -1, 1, 1, 1, 0, -1, 2],
                                 [0, 1, -1, 1, 1, 0, 1, 2],
                                 [1, 1, 1, 1, -1, 0, 1, 2]])

            self.switch_stencil_wall = []

            for side in range(6):
                self.opposite = []
                for i in range(len(self.e)):
                    for j in range(len(self.e)):
                        if self.e[i, self.s_a[side, 0]] == self.s_a[side, 1] and \
                                self.e[i, 0] == self.s_a[side, 2] * self.e[j, self.s_a[side, 3]] and \
                                self.e[i, 1] == self.s_a[side, 4] * self.e[j, self.s_a[side, 5]] and \
                                self.e[i, 2] == self.s_a[side, 6] * self.e[j, self.s_a[side, 7]]:
                            self.opposite.append((i, j))
                self.switch_stencil_wall.append(self.opposite)

            self.s_b = np.array([[0, -1, 1, 1, 0, -1, 1, 1, 2, 2],
                                 [0, 1, 1, -1, 0, 1, 1, -1, 2, 2],
                                 [0, 1, 1, 1, 0, -1, 1, -1, 2, 2],
                                 [0, -1, 1, -1, 0, 1, 1, 1, 2, 2],
                                 [0, -1, 2, 1, 0, 1, 2, -1, 1, 1],
                                 [1, -1, 2, 1, 1, 1, 2, -1, 0, 0],
                                 [1, 1, 2, 1, 0, -1, 2, -1, 0, 1],
                                 [0, 1, 2, 1, 1, -1, 2, -1, 1, 0],
                                 [1, 1, 2, -1, 1, -1, 2, 1, 0, 0],
                                 [0, -1, 2, -1, 1, 1, 2, 1, 1, 0],
                                 [1, -1, 2, -1, 0, 1, 2, 1, 0, 1],
                                 [0, 1, 2, -1, 0, -1, 2, 1, 1, 1]])

            self.switch_stencil_borders = []

            for b in range(12):
                self.opposite = []
                for i in range(len(self.e)):
                    for j in range(len(self.e)):
                        if self.e[i, self.s_b[b, 0]] == self.s_b[b, 1] and self.e[i, self.s_b[b, 2]] == self.s_b[b, 3] and \
                            self.e[j, self.s_b[b, 4]] == self.s_b[b, 5] and self.e[j, self.s_b[b, 6]] == self.s_b[b, 7] and \
                                self.e[i, self.s_b[b, 8]] == self.e[j, self.s_b[b, 9]]:
                            self.opposite.append((i, j))
                self.switch_stencil_borders.append(self.opposite)

            self.opposite = []
            self.switch_stencil_corner = []

            for i in range(len(self.e)):
                for j in range(len(self.e)):
                    if self.e[i, 0] != 0 and self.e[i, 1] != 0 and self.e[i, 2] != 0 and self.e[i, 0] == -self.e[j, 0] and \
                            self.e[i, 1] == -self.e[j, 1] and self.e[i, 2] == -self.e[j, 2]:
                        self.opposite.append((i, j))
            self.switch_stencil_corner.append(self.opposite)

        #####################################
        self.swap_w= [[(0,slice(None),slice(None)),(-1,slice(None),slice(None))]
                                             ,[(slice(None),0,slice(None)),(slice(None),-1,slice(None))]
                                             ,[(slice(None),slice(None),-1),(slice(None),slice(None),0)]
                                             ,[(slice(None),slice(None),0),(slice(None),slice(None),-1)]
                                             ,[(-1,slice(None),slice(None)),(slice(None),0,slice(None))]
                                             ,[(slice(None),-1,slice(None)),(0,slice(None),slice(None))]]

        self.f_copies=torch.stack((f[:,0,:,:].clone(),f[:,:,0,:].clone(),f[:,:,:,-1].clone(),f[:,:,:,0].clone(),f[:,-1,:,:].clone(),f[:,:,-1,:].clone()), dim=3)

        for i in range(6):
            for j in range(len(self.switch_stencil_wall[i])):

                if i == 3:
                    f[self.switch_stencil_wall[i][j][1],*self.swap_w[i][1]] = torch.transpose(self.f_copies[self.switch_stencil_wall[i][j][0],:, :, i],0,1)
                else:

                    f[self.switch_stencil_wall[i][j][1],*self.swap_w[i][1]]=self.f_copies[self.switch_stencil_wall[i][j][0],:,:,i]

        self.f_copies_borders=torch.stack((f[:,0,-1,:].clone(),f[:,-1,0,:].clone(),f[:,-1,-1,:].clone(),f[:,0,0,:].clone(),f[:,:,0,-1].clone()
                                           ,f[:,0,:,-1].clone(),f[:,:,-1,-1].clone(),f[:,-1,:,-1].clone(),f[:,:,-1,0].clone(),f[:,0,:,0].clone()
                                           ,f[:,:,0,0].clone(),f[:,-1,:,0].clone()),dim=2)

        self.borders=[(0,-1,slice(None)),(-1,0,slice(None)),(0,0,slice(None)),(-1,-1,slice(None)),(slice(None),-1,0)
                                  ,(-1,slice(None),0),(0,slice(None),0),(slice(None),0,0),(slice(None),0,-1),(slice(None),-1,-1),
            (slice(None),0,0),(-1,slice(None),0)]
        for i in range(12):
            for j in range(len(self.switch_stencil_borders[i])):

                f[self.switch_stencil_borders[i][j][0], *self.borders[i]] = self.f_copies_borders[self.switch_stencil_borders[i][j][1],:, i]

        return f