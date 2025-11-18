import torch
import numpy as np

from typing import List, Optional
from lettuce import Boundary, Flow, Context

__all__ = ["FullwayBounceBackBoundary"]

class FullwayBounceBackBoundary(Boundary):

    def __init__(self, context, flow, mask, global_solid_mask=None, periodicity: tuple[bool,...] = None, calc_force=None):
        self.context = context
        self.flow = flow

        self.mask = mask
        self.solid_mask = mask

        if periodicity is None:
            periodicity = (False, False, False if self.flow.stencil.d == 3 else None)

        # TODO: correct periodicity and solid-to-solid contact: periodicity attribute and global solid mask! in neighbor search
        # global_solid_mask to filter out all "fake" fluid neighbors, which are outside the FWBB but not in the fluid region
        if global_solid_mask is None:
            global_solid_mask = np.zeros_like(self.mask, dtype=bool)
        other_solid_bc_mask = np.where(~self.mask, global_solid_mask, False)  # exclude self.mask from global_solid_mask

        if calc_force is not None:
            self.force_sum = torch.zeros_like(self.context.convert_to_tensor(
                self.context.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.calc_force = True
        else:
            self.calc_force = False

        ### create f_mask_fwbb, needed for force-calculation
        # ...(marks all fs which streamed into the boundary in prior streaming step)
        # ... in other words: marks all fs that need to be bounced
        self.f_index_fwbb = []

        if self.flow.stencil.d == 2:
            nx, ny = mask.shape  # domain size in x and y

                # f_mask: [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
            ix_sp, iy_sp = np.where(mask)
                # np.arrays: list of (ix_sp) x-indizes and (iy_sp) y-indizes in the boundary.mask
                # ...to enable iteration over all boundary/wall/object-nodes
            for sp_index in range(0, len(ix_sp)):  # for all TRUE-nodes in boundary.mask
                for q_index in range(0, self.flow.stencil.q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.flow.stencil.d, dtype=int)
                    if ix_sp[sp_index] == 0 and self.context.stencil.e[q_index, 0] == -1 and periodicity[0]:  # searching border on left
                        border[0] = -1
                    elif ix_sp[sp_index] == nx - 1 and self.flow.stencil.e[q_index, 0] == 1 and periodicity[0]:  # searching border on right
                        border[0] = 1
                    if iy_sp[sp_index] == 0 and self.context.stencil.e[q_index, 1] == -1 and periodicity[1]:  # searching border on left
                        border[1] = -1
                    elif iy_sp[sp_index] == ny - 1 and self.flow.stencil.e[q_index, 1] == 1 and periodicity[1]:  # searching border on right
                        border[1] = 1
                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if (not mask[ix_sp[sp_index] + self.context.stencil.e[q_index, 0] - border[0]*nx,
                                    iy_sp[sp_index] + self.context.stencil.e[q_index, 1] - border[1]*ny]
                                and not other_solid_bc_mask[ix_sp[sp_index] + self.context.stencil.e[q_index, 0] - border[0]*nx,
                                    iy_sp[sp_index] + self.context.stencil.e[q_index, 1] - border[1]*ny]):
                            # if the neighbour of sp_index is False in the boundary.mask, sp_index is ix_sp solid node, neighbouring ix_sp fluid node:
                            # ...the direction pointing from the fluid neighbour to solid sp_index is marked on the solid sp_index

                            self.f_index_fwbb.append([self.flow.stencil.opposite[q_index], ix_sp[sp_index], iy_sp[sp_index]])  # list of [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.flow.stencil.d == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape

            ix_sp, iy_sp, iz_sp = np.where(mask)
            for sp_index in range(0, len(ix_sp)):
                for q_index in range(0, self.flow.stencil.q):
                    border = np.zeros(self.flow.stencil.d, dtype=int)
                    if ix_sp[sp_index] == 0 and self.context.stencil.e[q_index, 0] == -1 and periodicity[0]:  # searching border on left
                        border[0] = -1
                    elif ix_sp[sp_index] == nx - 1 and self.flow.stencil.e[q_index, 0] == 1 and periodicity[0]:  # searching border on right
                        border[0] = 1
                    if iy_sp[sp_index] == 0 and self.context.stencil.e[q_index, 1] == -1 and periodicity[1]:  # searching border on left
                        border[1] = -1
                    elif iy_sp[sp_index] == ny - 1 and self.flow.stencil.e[q_index, 1] == 1 and periodicity[1]:  # searching border on right
                        border[1] = 1
                    if iz_sp[sp_index] == 0 and self.context.stencil.e[q_index, 2] == -1 and periodicity[2]:  # searching border on left
                        border[2] = -1
                    elif iz_sp[sp_index] == nz - 1 and self.flow.stencil.e[q_index, 2] == 1 and periodicity[2]:  # searching border on right
                        border[2] = 1
                    try:  # try in case the neighboring cell does not exist (and f pointing out of simulation domain)
                        if (not mask[ix_sp[sp_index] + self.context.stencil.e[q_index, 0] - border[0] * nx,
                                    iy_sp[sp_index] + self.context.stencil.e[q_index, 1] - border[1] * ny,
                                    iz_sp[sp_index] + self.context.stencil.e[q_index, 2] - border[2] * nz]
                                and not other_solid_bc_mask[ix_sp[sp_index] + self.context.stencil.e[q_index, 0] - border[0] * nx,
                                                            iy_sp[sp_index] + self.context.stencil.e[q_index, 1] - border[1] * ny,
                                                            iz_sp[sp_index] + self.context.stencil.e[q_index, 2] - border[2] * nz]):
                            self.f_index_fwbb.append([self.flow.stencil.opposite[q_index], ix_sp[sp_index], iy_sp[sp_index], iz_sp[sp_index]])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.f_index_fwbb = torch.tensor(np.array(self.f_index_fwbb), device=self.flow.stencil.device, dtype=torch.int64) # the batch-index has to be integer
        #PHILIPP_occ_angepasst? self.f_index = torch.tensor(self.f_index, device=self.flow.stencil.device, dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.flow.stencil.opposite, device=self.flow.stencil.device,
                                            dtype=torch.int64)  # batch-index has to be ix_sp tensor


    def __call__(self, flow):
        # FULLWAY-BBBC: inverts populations on all boundary nodes

        # calc force on boundary:#
        if self.calc_force:
            self.calc_force_on_boundary(flow.f)
        # bounce (invert populations on boundary nodes)
        # f = torch.where(self.mask, f[self.flow.stencil.opposite], f)

        if self.flow.stencil.d == 2:
            flow.f[self.opposite_tensor[self.f_index_fwbb[:, 0]],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2]] = flow.f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2]]
        if self.flow.stencil.d == 3:
            flow.f[self.opposite_tensor[self.f_index_fwbb[:, 0]],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2],
            self.f_index_fwbb[:, 3]] = flow.f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2],
            self.f_index_fwbb[:, 3]]

    def calc_force_on_boundary(self, f):
        if self.flow.stencil.d == 2:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2]], self.flow.stencil.e[self.f_index_fwbb[:, 0]])
        if self.flow.stencil.d == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2],
            self.f_index_fwbb[:, 3]], self.flow.stencil.e[self.f_index_fwbb[:, 0]])

    def make_no_collision_mask(self, f_shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        assert self.mask.shape == f_shape[1:]
        return self.context.convert_to_tensor(self.mask)

    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        pass

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int):
        pass