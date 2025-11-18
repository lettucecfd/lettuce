import torch
import numpy as np

from typing import List, Optional
from lettuce import Boundary, Flow, Context
from solid_boundary_data import SolidBoundaryData

__all__ = ["HalfwayBounceBackBoundary"]

class HalfwayBounceBackBoundary(Boundary):

    def __init__(self, context, flow, solid_boundary_data: SolidBoundaryData = None, global_solid_mask=None, periodicity: tuple[bool,...] = None, calc_force=None):

        self.context = context
        self.flow = flow

        self.mask = solid_boundary_data.solid_mask
        self.solid_mask = solid_boundary_data.solid_mask

        # global_solid_mask to filter out all "fake" fluid neighbors, which are outside this HWBB but not in the fluid region
        if global_solid_mask is None:
            global_solid_mask = self.mask

        if periodicity is None:
            periodicity = (False, False, False if self.flow.context.d == 3 else None)

        if calc_force is not None:
            self.force_sum = torch.zeros_like(self.context.convert_to_tensor(
                self.flow.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.calc_force = True
        else:
            self.calc_force = False

        self.f_index = []
        self.f_collided = None

        # combine f_index_lt and f_index_gt to self.f_index
        if (hasattr(solid_boundary_data, "f_index_gt") or hasattr(solid_boundary_data, "f_index_lt")) and len(solid_boundary_data.f_index_lt.shape) == len(solid_boundary_data.f_index_gt.shape):  # if solid_boundary_data contains batch_indices, use them
            self.f_index = np.concatenate((solid_boundary_data.f_index_lt, solid_boundary_data.f_index_gt), axis=0)
        elif hasattr(solid_boundary_data, "f_index_gt") and solid_boundary_data.f_index_lt.shape[0] == 0:
            self.f_index = solid_boundary_data.f_index_gt
        elif hasattr(solid_boundary_data, "f_index_lt") and solid_boundary_data.f_index_gt.shape[0] == 0:
            self.f_index = solid_boundary_data.f_index_lt
        else:  #else do ghetto-neighbour_search below
            print("(INFO) HWBB didn't find solid_boundary_data, doing legacy neighbour_search on mask...")
            # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
            if self.flow.context.d == 2:
                nx, ny = self.mask.shape  # domain size in x and y
                a, b = np.where(self.mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

                for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                    for i in range(0, self.flow.stencil.q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                        # check for boundary-nodes neighboring the domain-border.
                        # ...they have to take the periodicity into account...
                        border = np.zeros(self.flow.context.d, dtype=int)

                        if a[p] == 0 and self.flow.stencil.e[i, 0] == -1 and periodicity[0]:  # searching border on left [x]
                            border[0] = -1
                        elif a[p] == nx - 1 and self.flow.stencil.e[i, 0] == 1 and periodicity[0]:  # searching border on right [x]
                            border[0] = 1

                        if b[p] == 0 and self.flow.stencil.e[i, 1] == -1 and periodicity[1]:  # searching border on left [y]
                            border[1] = -1
                        elif b[p] == ny - 1 and self.flow.stencil.e[i, 1] == 1 and periodicity[1]:  # searching border on right [y]
                            border[1] = 1

                        try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                            if (not self.mask[a[p] + self.flow.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.flow.stencil.e[i, 1] - border[1] * ny]
                                and not global_solid_mask[
                                    a[p] + self.flow.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.flow.stencil.e[i, 1] - border[1] * ny]):
                                # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                                # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                                self.f_index.append([self.flow.stencil.opposite[i],
                                                     a[p] + self.flow.stencil.e[i, 0] - border[0] * nx,
                                                     b[p] + self.flow.stencil.e[i, 1] - border[1] * ny])
                        except IndexError:
                            pass  # just ignore this iteration since there is no neighbor there

            if self.flow.context.d == 3:  # like 2D, but in 3D...guess what...
                nx, ny, nz = self.mask.shape
                a, b, c = np.where(self.mask)

                for p in range(0, len(a)):
                    for i in range(0, self.flow.stencil.q):
                        border = np.zeros(self.flow.context.d, dtype=int)
                        # x - direction
                        if a[p] == 0 and self.flow.stencil.e[i, 0] == -1 and periodicity[0]:  # searching border on left
                            border[0] = -1
                        elif a[p] == nx - 1 and self.flow.stencil.e[i, 0] == 1 and periodicity[0]:  # searching border on right
                            border[0] = 1
                        # y - direction
                        if b[p] == 0 and self.flow.stencil.e[i, 1] == -1 and periodicity[1]:  # searching border on left
                            border[1] = -1
                        elif b[p] == ny - 1 and self.flow.stencil.e[i, 1] == 1 and periodicity[1]:  # searching border on right
                            border[1] = 1
                        # z - direction
                        if c[p] == 0 and self.flow.stencil.e[i, 2] == -1 and periodicity[2]:  # searching border on left
                            border[2] = -1
                        elif c[p] == nz - 1 and self.flow.stencil.e[i, 2] == 1 and periodicity[2]:  # searching border on right
                            border[2] = 1

                        try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                            if (not self.mask[a[p] + self.flow.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.flow.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.flow.stencil.e[i, 2] - border[2] * nz]
                                and not global_solid_mask[
                                    a[p] + self.flow.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.flow.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.flow.stencil.e[i, 2] - border[2] * nz]):

                                    self.f_index.append([self.flow.stencil.opposite[i],
                                                         a[p] + self.flow.stencil.e[i, 0] - border[0] * nx,
                                                         b[p] + self.flow.stencil.e[i, 1] - border[1] * ny,
                                                         c[p] + self.flow.stencil.e[i, 2] - border[2] * nz])
                        except IndexError:
                            pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_index = torch.tensor(np.array(self.f_index), device=self.flow.context.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.flow.stencil.opposite, device=self.flow.context.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor
        # f_collided = torch.zeros_like(self.f_index[:, 0], dtype=self.flow.context.dtype)
        # f_collided_opposite = torch.zeros_like(self.f_index[:, 0], dtype=self.flow.context.dtype)
        # self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)

    def __call__(self, flow):
        # calc force on boundary:
        if self.calc_force:
            self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.flow.stencil.opposite], f_collided[self.flow.stencil.opposite], f)

        if self.flow.context.d == 2:
            flow.f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = self.f_collided[:, 0]
        if self.flow.context.d == 3:
            flow.f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = self.f_collided[:, 0]

    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_stream_mask
        # ...prevents interaction between nodes inside and outside the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post-processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == shape[1:]
        return self.context.convert_to_tensor(self.mask)

    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        assert self.mask.shape == shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.context.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
        # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
        # ...populations pointing at the surface of the boundary
        self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided[:, 0], self.flow.stencil.e[self.f_index[:, 0]])

    #TODO: find a way to use pre- and post-Streaming Populations for bounce...
    def store_f_collided(self, f_collided):
        if self.flow.context.d == 2:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
            self.f_index[:, 1],  # x
            self.f_index[:, 2]])  # y
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:, 0]],  # q
            self.f_index[:, 1],  # x
            self.f_index[:, 2]])  # y
        if self.flow.context.d == 3:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
            self.f_index[:, 1],  # x
            self.f_index[:, 2],  # y
            self.f_index[:, 3]])  # z
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:, 0]],  # q
            self.f_index[:, 1],  # x
            self.f_index[:, 2],  # y
            self.f_index[:, 3]])  # z

    def initialize_f_collided(self):
        f_collided = torch.zeros_like(self.f_index[:, 0], dtype=self.flow.context.dtype)
        f_collided_opposite = torch.zeros_like(self.f_index[:, 0], dtype=self.flow.context.dtype)
        self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)
        
    def native_available(self) -> bool:
        return True

    def native_generator(self, index: int):
        pass