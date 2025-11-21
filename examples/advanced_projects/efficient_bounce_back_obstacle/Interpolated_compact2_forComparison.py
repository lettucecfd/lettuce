import numpy as np
import torch

class InterpolatedBounceBackBoundary_compact_v2:

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))

        self.f_index_lt = []  # indices of relevant populations (for bounce back and force-calculation) with d<=0.5
        self.f_index_gt = []  # indices of relevant populations (for bounce back and force-calculation) with d>0.5
        self.d_lt = []  # distances between node and boundary for d<0.5
        self.d_gt = []  # distances between node and boundary for d>0.5

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)

                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            a, b, c = np.where(mask)

            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1

                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            # Z-coodinate not needed for cylinder !

                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node
                            # Z-coodinate not needed for cylinder !

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,z,ci", a[p],
                                      b[p], c[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_index_lt = torch.tensor(np.array(self.f_index_lt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(np.array(self.f_index_gt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.d_lt = self.lattice.convert_to_tensor(np.array(self.d_lt))
        self.d_gt = self.lattice.convert_to_tensor(np.array(self.d_gt))
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

        f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)

        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")

    def __call__(self, f):
        ## f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.lattice.D == 2:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                                                     self.f_index_lt[:, 1],
                                                                                                     self.f_index_lt[:, 2]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        if self.lattice.D == 3:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2],
              self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                                                     self.f_index_lt[:, 1],
                                                                                                     self.f_index_lt[:, 2],
                                                                                                     self.f_index_lt[:, 3]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2],
              self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]


        # CALC. FORCE on boundary (MEM, MEA)
        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opp.])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2]],
                                            self.lattice.e[self.f_index_gt[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2],
                                              self.f_index_lt[:, 3]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2],
                                                self.f_index_gt[:, 3]],
                                            self.lattice.e[self.f_index_gt[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2]])  # y
            self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2]])  # y

            self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                          self.f_index_gt[:, 1],  # x
                                                          self.f_index_gt[:, 2]])  # y
            self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:,0]],  # q
                                                          self.f_index_gt[:, 1],  # x
                                                          self.f_index_gt[:, 2]])  # y
        if self.lattice.D == 3:
            self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2],  # y
                                                          self.f_index_lt[:, 3]])  # z
            self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2],  # y
                                                          self.f_index_lt[:, 3]])  # z

            self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                          self.f_index_gt[:, 1],  # x
                                                          self.f_index_gt[:, 2],  # y
                                                          self.f_index_gt[:, 3]])  # z
            self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:,0]],  # q
                                                          self.f_index_gt[:, 1],  # x
                                                          self.f_index_gt[:, 2],  # y
                                                          self.f_index_gt[:, 3]])  # z