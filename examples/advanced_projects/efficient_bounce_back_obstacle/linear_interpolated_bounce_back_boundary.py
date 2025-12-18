import torch

from lettuce import Boundary, Flow, Context
from solid_boundary_data import SolidBoundaryData

__all__ = ["LinearInterpolatedBounceBackBoundary"]

class LinearInterpolatedBounceBackBoundary(Boundary):
    """Interpolated Bounce Back Boundary Condition (IBB or IBB1)
       first introduced by Bouzidi et al. (2001), as described in Kruger et al. (2017)
        - linear interpolation of populations to retain the true boundary location between fluid-
        and solid-node
        - improved accuracy in comparison to FWBB and HWBB
        - able to calculate fluid force on boundary by momentum exchange
        - (!) RELIES on the SolidBoundaryData object to be provided,
            which supplies the f_index and interpolation constants.
    """

    def __init__(self, context, flow, solid_boundary_data: SolidBoundaryData, calc_force: bool =False):
        self.context = context
        self.flow = flow

        self.mask = solid_boundary_data.solid_mask

        if calc_force:
            # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.force_sum = torch.zeros_like(self.context.convert_to_tensor(
                self.flow.stencil.e[0]))
            self.calc_force = True
        else:
            self.calc_force = False

        # convert relevant data to tensors:
        ### TODO (QUESTION): fix batch-index-datatype (integer)?
        self.f_index_lt = torch.tensor(solid_boundary_data.f_index_lt,
                                       device=self.context.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(solid_boundary_data.f_index_gt,
                                       device=self.context.device,
                                       dtype=torch.int64)  # the batch-index has to be integer

        self.d_lt = self.context.convert_to_tensor(solid_boundary_data.d_lt)
        self.d_gt = self.context.convert_to_tensor(solid_boundary_data.d_gt)
        self.opposite_tensor = torch.tensor(self.flow.stencil.opposite,
                                            device=self.context.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor
        # TODO (optional): replace self.opposite_tensor with flow.torch_stencil.opposite

        self.f_collided_lt = None
        self.f_collided_gt = None
        # will be populated in initialize_f_collided() method


    def __call__(self, flow):
        """ IBB1: interpolate bounced populations from two fluid nodes"""
        ## reminder: f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.flow.stencil.d == 2:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                flow.f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2]] = (2 * self.d_lt * self.f_collided_lt[:, 0]
                                          + (1 - 2 * self.d_lt) * flow.f[
                                self.f_index_lt[:, 0],
                                self.f_index_lt[:, 1],
                                self.f_index_lt[:, 2]])
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                flow.f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2]] = ((1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0]
                                          + (1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1])

        if self.flow.stencil.d == 3:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                flow.f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2],
                self.f_index_lt[:, 3]] = (2 * self.d_lt * self.f_collided_lt[:, 0]
                                          + (1 - 2 * self.d_lt) * flow.f[
                                self.f_index_lt[:, 0],
                                self.f_index_lt[:, 1],
                                self.f_index_lt[:, 2],
                                self.f_index_lt[:, 3]])
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                flow.f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2],
                self.f_index_gt[:, 3]] = ((1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0]
                                          + (1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1])

        # CALC. FORCE on boundary (MEM, MEA)
        if self.calc_force:
            self.calc_force_on_boundary(flow.f)

    def make_no_streaming_mask(self, f_shape, context: Context):
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional),
        # but CAN be (x,y,z) (z optional).
        # in the latter case, torch.where broadcasts the mask to (q,x,y,z),
        # so ALL q populations of a lattice-node are marked equally
        return self.context.convert_to_tensor(self.mask, dtype=bool)

    def make_no_collision_mask(self, f_shape, context: Context):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...)
        # on the boundary nodes, in the initial solution of your flow,
        # especially if visualization or post-processing uses the field-values
        # in the whole domain (including the boundary region)!
        return self.context.convert_to_tensor(self.mask, dtype=bool)

    def calc_force_on_boundary(self, f_bounced):
        """calulate the fluid force on the boundary by Momentum Exchange"""
        ### basically: force = e * (f_collided + f_bounced[opp.])
        if self.flow.stencil.d == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2]],
                                          self.flow.torch_stencil.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2]],
                                            self.flow.torch_stencil.e[self.f_index_gt[:, 0]])
        if self.flow.stencil.d == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2],
                                              self.f_index_lt[:, 3]],
                                          self.flow.torch_stencil.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2],
                                                self.f_index_gt[:, 3]],
                                            self.flow.torch_stencil.e[self.f_index_gt[:, 0]])


    def store_f_collided(self, f_collided):
        """store populations between collision and streaming, because they are
                    needed for calculation of bounce and force!"""
        if self.flow.stencil.d == 2:
            if len(self.f_collided_lt) != 0:
                self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2]])  # y
                self.f_collided_lt[:, 1] = (
                    torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2]]))  # y
            if len(self.f_collided_gt) != 0:
                self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2]])  # y
                self.f_collided_gt[:, 1] = (
                    torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:,0]],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2]]))  # y
        if self.flow.stencil.d == 3:
            if len(self.f_collided_lt) != 0:
                self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2],  # y
                                                              self.f_index_lt[:, 3]])  # z
                self.f_collided_lt[:, 1] = (
                    torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2],  # y
                                                              self.f_index_lt[:, 3]]))  # z
            if len(self.f_collided_gt) != 0:
                self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                                  self.f_index_gt[:, 1],  # x
                                                                  self.f_index_gt[:, 2],  # y
                                                                  self.f_index_gt[:, 3]])  # z
                self.f_collided_gt[:, 1] = (
                    torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:, 0]],  # q
                                                                  self.f_index_gt[:, 1],  # x
                                                                  self.f_index_gt[:, 2],  # y
                                                                  self.f_index_gt[:, 3]]))  # z

    def initialize_f_collided(self):
        """initialize the tensors to store post-collision populations"""

        # float-tensor with number of (x_b nodes with d<=0.5) values
        f_collided_lt = torch.zeros_like(self.d_lt)

        # float-tensor with number of (x_b nodes with d>0.5) values
        f_collided_gt = torch.zeros_like(self.d_gt)

        f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int):
        # not implemented yet
        return None
