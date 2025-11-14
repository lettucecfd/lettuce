import torch
import numpy as np

from typing import List, Optional
from lettuce.lettuce import Boundary, Flow, Context
from lettuce.lettuce.ext import SolidBoundaryData

__all__ = ["LinearInterpolatedBounceBackBoundary"]




# THIS IS THE IBB in compact implementation WITH OCC-support
class LinearInterpolatedBounceBackBoundary(Boundary):
    """Interpolated Bounce Back Boundary Condition first introduced by Bouzidi et al. (2001), as described in Kruger et al.
        (2017)
        - linear or quadratic interpolation of populations to retain the true boundary location between fluid- and
        solid-node
        * version 2.0: using given indices and distances between fluid- and solid-node
        of boundary link and boundary surface for interpolation!
    """

    def __init__(self, context, flow, solid_boundary_data: SolidBoundaryData, calc_force=None):
        self.context = context
        self.flow = flow

        self.mask = solid_boundary_data.solid_mask
        self.solid_mask = solid_boundary_data.solid_mask
        if calc_force is not None:
            self.force_sum = torch.zeros_like(self.context.convert_to_tensor(
                self.flow.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.calc_force = True
        else:
            self.calc_force = False

        # convert relevant tensors:
        ### TODO: fix batch-index-datatype...?
        self.f_index_lt = torch.tensor(solid_boundary_data.f_index_lt, device=self.context.device, dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(solid_boundary_data.f_index_gt, device=self.context.device, dtype=torch.int64)  # the batch-index has to be integer

        self.d_lt = self.context.convert_to_tensor(solid_boundary_data.d_lt)
        self.d_gt = self.context.convert_to_tensor(solid_boundary_data.d_gt)
        self.opposite_tensor = torch.tensor(self.flow.stencil.opposite, device=self.context.device, dtype=torch.int64)  # batch-index has to be a tensor

        self.f_collided_lt = None
        self.f_collided_gt = None
        # will be populated in initialize_f_collided() method

        # DEPRECATED: got moved to method initialize f_collided
        # f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        # f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        # f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        # f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        # self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        # self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)
        #TODO: does the f_collided etc. have to be reworked due to new lettuce master pre/post collision boundary stuff?

    def __call__(self, flow):
        ## f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.flow.stencil.d == 2:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                flow.f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * flow.f[
                    self.f_index_lt[:, 0],
                    self.f_index_lt[:, 1],
                    self.f_index_lt[:, 2]]
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                flow.f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        if self.flow.stencil.d == 3:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                flow.f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2],
                self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * flow.f[
                    self.f_index_lt[:, 0],
                    self.f_index_lt[:, 1],
                    self.f_index_lt[:, 2],
                    self.f_index_lt[:, 3]]
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                flow.f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2],
                self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        # CALC. FORCE on boundary (MEM, MEA)
        if self.calc_force:
            self.calc_force_on_boundary(flow.f)

    def make_no_streaming_mask(self, shape, context: Context):
        assert self.mask.shape == shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        # return torch.tensor(self.mask, dtype=torch.bool)
        return self.context.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, shape, context: Context):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post-processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == shape[1:]
        # return torch.tensor(self.mask, dtype=torch.bool)  # self.context.convert_to_tensor(self.mask)
        return self.context.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opp.])
        if self.flow.stencil.d == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2]],
                                          self.flow.stencil.e[self.f_index_lt[:, 0]].float()) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2]],
                                            self.flow.stencil.e[self.f_index_gt[:, 0]].float())
        if self.flow.stencil.d == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2],
                                              self.f_index_lt[:, 3]],
                                          self.flow.stencil.e[self.f_index_lt[:, 0]].float()) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2],
                                                self.f_index_gt[:, 3]],
                                            self.flow.stencil.e[self.f_index_gt[:, 0]].float())

    # TODO: find a way to use pre- and post-Streaming Populations for bounce...
    def store_f_collided(self, f_collided):
        for f_index_lgt, f_collided_lgt in zip([self.f_index_lt, self.f_index_gt],
                                               [self.f_collided_lt, self.f_collided_gt]):
            if len(f_index_lgt) != 0:
                for d in range(self.flow.stencil.d):
                    indices = [f_index_lgt[:, 0],  # q
                               f_index_lgt[:, 1],  # x
                               f_index_lgt[:, 2]]  # y
                    if self.flow.stencil.d == 3:
                        indices.append(f_index_lgt[:, 3])
                    f_collided_lgt[:, 0] = torch.clone(f_collided[indices])
                    indices[0] = self.opposite_tensor[f_index_lgt[:, 0]]
                    f_collided_lgt[:, 1] = torch.clone(f_collided[indices])
        # TODO: compare performance of THIS to original hardcoded "store_f_collided()" of IBB1, see below

    # >>> OLD version "semi hardcoded"
    # def store_f_collided(self, f_collided):
    #     if self.flow.stencil.d == 2:
    #         if len(self.f_collided_lt) != 0:
    #             self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2]])  # y
    #             self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2]])  # y
    #         if len(self.f_collided_gt) != 0:
    #             self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
    #                                                           self.f_index_gt[:, 1],  # x
    #                                                           self.f_index_gt[:, 2]])  # y
    #             self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:,0]],  # q
    #                                                           self.f_index_gt[:, 1],  # x
    #                                                           self.f_index_gt[:, 2]])  # y
    #     if self.flow.stencil.d == 3:
    #         if len(self.f_collided_lt) != 0:
    #             self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2],  # y
    #                                                           self.f_index_lt[:, 3]])  # z
    #             self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2],  # y
    #                                                           self.f_index_lt[:, 3]])  # z
    #         if len(self.f_collided_gt) != 0:
    #             self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
    #                                                               self.f_index_gt[:, 1],  # x
    #                                                               self.f_index_gt[:, 2],  # y
    #                                                               self.f_index_gt[:, 3]])  # z
    #             self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:, 0]],  # q
    #                                                               self.f_index_gt[:, 1],  # x
    #                                                               self.f_index_gt[:, 2],  # y
    #                                                               self.f_index_gt[:, 3]])  # z
    # <<< OLD version "semi hardcoded"

    # TODO: find a way to use pre- and post-Streaming Populations for bounce...
    def initialize_f_collided(self):
        f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int):
        return None
