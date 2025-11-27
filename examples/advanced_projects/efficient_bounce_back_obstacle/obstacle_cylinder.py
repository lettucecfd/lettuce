import warnings
from typing import Union, List, Optional

import numpy as np
import torch

from lettuce.ext._flows import ExtFlow
from lettuce import UnitConversion, Context, Stencil, Equilibrium
from lettuce.util import append_axes
from lettuce.ext import (EquilibriumBoundaryPU, BounceBackBoundary,
                                 EquilibriumOutletP, AntiBounceBackOutlet)
from examples.advanced_projects.efficient_bounce_back_obstacle import SolidBoundaryData, FullwayBounceBackBoundary, HalfwayBounceBackBoundary, LinearInterpolatedBounceBackBoundary

__all__ = ['ObstacleCylinder']


class ObstacleCylinder(ExtFlow):
    """
        unified version of 2D and 3D cylinder flow
        refined version of flow/obstacle.py, for cylinder-flow in 2D or 3D. The dimensions will be assumed from
        len(resolution)

        Flow:
        - inflow (EquilibriumBoundaryPU) at x=0, outflow (EquilibriumOutletP) at x=xmax
        - further boundaries depend on parameters:
            - lateral (y direction): periodic, no-slip wall, slip wall
            - lateral (z direction): periodic (only if lattice.D==3)
        - obstacle: cylinder obstacle centered at (y_lu/2+x_offset, y_LU/2+y_offset),
            with radius=char_length, uniform symmetry in z-direction
        - boundary condition for obstacle can be chosen: hwbb, fwbb, ibb1
            AND storage-formats: fwbb, fwbbc (compact), hwbb, hwbbc1, hwbbc2, hwbbc3, ibb1, ibb1c1, ibb1c2
            - c: compact implementation (= DIY-sparse, faster and less memory)
            - c2 better than c1
            - c3 is hwbb-only for use with older simulation-classes that don't contain a store_f_collided call!
            recommendation: use fwbbc, hwbbc2, ibb1c2 for best runtime- and memory-efficiency
        - initial pertubation (trigger Von Kármán vortex street for Re>46) can be initialized in y and z direction
        - initial velocity can be 0, u_char or a parabolic profile (parabolic if lateral_walls = "bounceback")
        - inlet/inflow velocity can be uniform u_char or parabolic

        Parameters:
        ----------
        <to fill>
        ----------
    """
    # TODO: rewrite description above

    def __init__(self, context: Context, resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 char_length_pu, char_length_lu, char_velocity_pu=1,
                 lateral_walls='periodic', bc_type='fwbb',
                 perturb_init=True, u_init=0,
                 x_offset=0, y_offset=0,
                 calc_force_coefficients=False,
                 stencil: Optional[Stencil] = None,
                 equilibrium: Optional[Equilibrium] = None):

        self.char_length_pu = char_length_pu  # characteristic length
        self.char_length_lu = char_length_lu # characteristic length in PU
        # TODO: praktische Dopplung wenn char_length_lu UND resolution zusammen mit char_length_pu angegeben werden...
        self.resolution = self.make_resolution(resolution, stencil) # shape in LU, if only INT, a cube shaped domain is assumed
        self.char_velocity_pu = char_velocity_pu

        print("OBST_Cylinder: self.resolution=", self.resolution)


        # flow and boundary settings
        self.perturb_init = perturb_init    # toggle: introduce asymmetry in initial solution to trigger v'Karman Vortex Street
        self.u_init = u_init                # toggle: initial solution velocity profile type
        self.lateral_walls = lateral_walls  # toggle: lateral walls to be bounce back (bounceback), slip wall (slip) or periodic (periodic)
        self.bc_type = bc_type              # toggle: bounce back algorithm: halfway (hwbb),  fullway (fwbb), linearly interpolated (ibb1)

        self.calc_force_coefficients = calc_force_coefficients
        self.periodicity = (False, False, True if len(self.resolution) == 3 else None)

        # initialize masks (init with zeros)
        self.solid_mask = np.zeros(shape=self.resolution, dtype=bool)  # marks all solid nodes (obstacle, walls, ...)
        self.wall_mask = np.zeros_like(self.solid_mask)                # marks lateral (top+bottom) walls
        self._obstacle_mask = np.zeros_like(self.solid_mask)           # marks all obstacle nodes (for fluid-solid-force_calc.)

        # initialize super class with unit conversion, equilibrium, context etc.
        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)
        # UnitConversion: defined below unter make_units(), executed by ExtFlow; flow object gets units-attibute!

        self.in_mask = np.zeros(self.grid[0].shape, dtype=bool)       # marks all inlet nodes


        # cylinder geometry in LU
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.radius_lu = char_length_lu / 2
        #todo: das PLUS hier ist ein Problem!
        self.y_pos_lu = self.resolution[1] / 2 + 0.5 + self.y_offset  # y_position of cylinder-center in 1-based indexing
        self.x_pos_lu = self.y_pos_lu + self.x_offset  # keep symmetry of cylinder in x and y direction

        # MESHGRID of x, y, (z) index (LU)
        xyz = tuple(np.linspace(1, n, n) for n in self.resolution)  # Tupel of index-lists (1-n (one-based!))
        if self.stencil.d == 2:
            x_lu, y_lu = np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y-index
        elif self.stencil.d == 3:
            x_lu, y_lu, z_lu = np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- and z-index
        else:
            x_lu, y_lu, z_lu = 1,1,1
            print("WARNING: something went wrong in LU-gird-index generation, lattice.D must be 2 or 3!")

        # BASIC mask-Version of circular cylinder.
        #TODO: Options for obstacle def: 1) basic mask (no interpolation), 2) IBB-index and mask, 3) OCC-enabled
        condition = np.sqrt((x_lu - self.x_pos_lu) ** 2 + (y_lu - self.y_pos_lu) ** 2) < self.radius_lu
        self.obstacle_mask[np.where(condition)] = 1
        self.solid_mask[np.where(condition)] = 1

        print("OBST_Cylinder: x_pos, y_pos, radius:", self.x_pos_lu, self.y_pos_lu, self.radius_lu)

        # MASKS for solid boundaries (lateral walls, obstacle, solid (= obstacle and walls), inlet)
        # (INFO): indexing doesn't need z-Index for 3D, everything is broadcasted along z!
        if self.lateral_walls == 'bounceback' or self.lateral_walls == 'slip':  # if top and bottom are link-based BC
            self.wall_mask[:, [0, -1]] = True  # don't mark wall nodes as inlet
            self.solid_mask[np.where(self.wall_mask)] = 1  # mark solid walls
            self.in_mask[0, 1:-1] = True  # inlet on the left, except for top and bottom wall (y=0, y=y_max)
        else:  # if lateral_wals == 'periodic', no walls
            self.in_mask[0, :] = True  # inlet on the left (x=0)

        # generate parabolic velocity profile for inlet BC if lateral_walls (top and bottom) are bounce back walls (== channel-flow)
        self.u_inlet = self.units.characteristic_velocity_pu * self._unit_vector()  # u = [ux,uy,uz] = [1,0,0] in PU // uniform characteristic velocity in x-direction
        if self.lateral_walls == 'bounceback':
            ## parabolic velocity profile, zeroing on the edges
            ## How to parabola:
            ## 1.parabola in factorized form (GER: "Nullstellenform"): y = (x-x1)*(x-x2)
            ## 2.parabola with a maximum and zero at x1=0 und x2=x0: y=-x*(x-x0)
            ## 3.scale parabola, to make y_s(x_s)=1 the maximum: y=-x*(x-x0)*(1/(x0/2)²)
            ## (4. optional) scale amplitude with 1.5 to have a mean velocity of 1, also making the integral of a homogeneous velocity profile with u=1 and the parabolic profile being equal
            (nx, ny, nz) = self.resolution  # number of gridpoints in y direction
            parabola_y = np.zeros((1, ny))
            y_coordinates = np.linspace(0, ny, ny)  # linspace() creates n points between 0 and ny, including 0 and ny:
            # top and bottom velocity values will be zero to agree with wall-boundary-condition
            parabola_y[:, 1:-1] = - 1.5 * np.array(self.u_inlet).max() * y_coordinates[1:-1] * (
                        y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2  # parabolic velocity profile
            # scale with 1.5 to achieve a mean velocity of u_char! -> DIFFERENT FROM cylinder2D and cylinder3D (!)
            if self.stencil.d == 2:
                # in 2D u1 needs Dimension 1 x ny (!)
                velocity_y = np.zeros_like(parabola_y)  # y-velocities = 0
                self.u_inlet = np.stack([parabola_y, velocity_y], axis=0)  # stack/pack u-field
            elif self.stencil.d == 3:
                ones_z = np.ones(nz)
                parabola_yz = parabola_y[:, :, np.newaxis] * ones_z
                parabola_yz_zeros = np.zeros_like(parabola_yz)
                # create u_xyz inlet yz-plane:
                self.u_inlet = np.stack([parabola_yz, parabola_yz_zeros, parabola_yz_zeros], axis=0)  # stack/pack u-field

    def make_units(self, reynolds_number, mach_number, resolution: List[int]
                   ) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=self.char_length_lu,
            characteristic_length_pu=self.char_length_pu,
            characteristic_velocity_pu=self.char_velocity_pu
        )

    # gibt immer N-dimensional zurück! (flow.resolution darf auch ein int sein, dann wird quadratisch/kubisch angenommen)
    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * (stencil.d or self.stencil.d)
        else:
            return resolution

    @property
    def obstacle_mask(self):
        return self._obstacle_mask

    @obstacle_mask.setter
    def obstacle_mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.resolution
        self._obstacle_mask = m.astype(bool)

    def initial_pu(self):
        p = np.zeros_like(self.grid[0], dtype=float)[None, ...]
        u_max_pu = self.units.characteristic_velocity_pu * self._unit_vector()
        u_max_pu = append_axes(u_max_pu, self.stencil.d)
        self.solid_mask[np.where(self.obstacle_mask)] = 1  # TODO: is this line needed? OLD: # This line is needed, because the obstacle_mask.setter does not define the solid_mask properly (see above) #OLD
        ### initial velocity field: "u_init"-parameter
        # 0: uniform u=0
        # 1: uniform u=1 or parabolic (depends on lateral_walls -> bounceback => parabolic; slip, periodic => uniform)
        u = (1 - self.solid_mask) * u_max_pu
        if self.u_init == 0:
            u = u * 0  # uniform u=0
        else:
            if self.lateral_walls == 'bounceback':  # parabolic along y, uniform along x and z (similar to poiseuille-flow)
                ny = self.resolution[1]  # number of gridpoints in y direction
                ux_factor = np.zeros(ny)  # vector for one column (u(x=0))
                # multiply parabolic profile with every column of the velocity field:
                y_coordinates = np.linspace(0, ny, ny)
                ux_factor[1:-1] = - y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2
                if self.stencil.d == 2:
                    u = np.einsum('k,ijk->ijk', ux_factor, u)
                elif self.stencil.d == 3:
                    u = np.einsum('k,ijkl->ijkl', ux_factor, u)
            else:  # lateral_walls == periodic or slip
                # initial velocity u_PU=1 on every fluid node
                u = (1 - self.solid_mask) * u_max_pu

        ### perturb initial velocity field-symmetry (in y and z) to trigger 'von Karman' vortex street
        if self.perturb_init:  # perturb initial solution in y
            # overlays a sine-wave on the second column of nodes x_lu=1 (index 1)
            ny = self.grid[1].shape[1]
            if u.max() < 0.5 * self.units.characteristic_velocity_pu:
                # add perturbation for small velocities
                amplitude_y = np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * self.units.characteristic_velocity_pu * 0.1
                if self.stencil.d == 2:
                    u[0][1] += amplitude_y
                elif self.stencil.d == 3:
                    nz = self.grid[2].shape[2]
                    plane_yz = np.ones_like(u[0, 1])  # plane of ones
                    u[0][1] = np.einsum('y,yz->yz', amplitude_y, plane_yz)  # plane of amplitude in y
                    amplitude_z = np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * self.units.characteristic_velocity_pu * 0.1  # amplitude in z
                    u[0][1] += np.einsum('z,yz->yz', amplitude_z, plane_yz)
            else:
                # multiply scaled down perturbation if velocity field is already near u_char
                factor = 1 + np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * 0.1
                if self.stencil.d == 2:
                    u[0][1] *= factor
                elif self.stencil.d == 3:
                    nz = self.grid[2].shape[1]
                    plane_yz = np.ones_like(u[0, 1, :, :])
                    u[0][1] = np.einsum('y,yz->yz', factor, u[0][1])
                    factor = 1 + np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * 0.1  # perturbation in z-direction
                    u[0][1] = np.einsum('z,yz->yz', factor, u[0][1])
        return p, u

    def make_solid_boundary_data(self, x_center, y_center, radius):
        pass
        # NOT YET IMPLEMENTED; OCC-version for index lists
        # TODO: wo sollte das SolidBoundaryData Objekt definiert werden bzw. die Klasse definiert werden?

        # cylinder definitions:
            # OPTION 1: calculate directly from analytic function -> see make_ibb_index_lists below
            # OPTION 2: calculate via OCC (Philipp) -> see external house/MA
            # OPTION 3 (FWBB, HWBB only): calculate from mask or only give mask...

    def make_ibb_index_lists(self, x_center, y_center, radius):

        # this method relies on self.obstacle_mask too!
        f_index_lt = []  # indices of relevant populations (for bounce back and force-calculation) with d<=0.5
        f_index_gt = []  # indices of relevant populations (for bounce back and force-calculation) with d>0.5
        d_lt = []  # distances between node and boundary for d<0.5
        d_gt = []  # distances between node and boundary for d>0.5

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.stencil.d == 2:
            # TODO: the 2D and 3D options could be condensed/unified
            nx, ny = self.obstacle_mask.shape  # domain size in x and y
            a, b = np.where(self.obstacle_mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.stencil.q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.stencil.d, dtype=int)

                    if a[p] == 0 and self.stencil.e[i][0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.stencil.e[i][0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.stencil.e[i][1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.stencil.e[i][1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not self.obstacle_mask[a[p] + self.stencil.e[i][0] - border[0] * nx,
                        b[p] + self.stencil.e[i][1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.stencil.e[i][0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.stencil.e[i][1] - border[1] * ny  # fluid node y-coordinate
                            cx = self.stencil.e[self.stencil.opposite[i]][ 0]  # link-direction x to solid node
                            cy = self.stencil.e[self.stencil.opposite[i]][ 1]  # link-direction y to solid node

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
                                    d_lt.append(d1)
                                    f_index_lt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny])
                                else:  # d>0.5
                                    d_gt.append(d1)
                                    f_index_gt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    d_lt.append(d2)
                                    f_index_lt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny])
                                else:  # d>0.5
                                    d_gt.append(d2)
                                    f_index_gt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], self.stencil.e[i][0], self.stencil.e[i][1])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.stencil.d == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = self.obstacle_mask.shape
            a, b, c = np.where(self.obstacle_mask)

            for p in range(0, len(a)):
                for i in range(0, self.stencil.q):
                    border = np.zeros(self.stencil.d, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.stencil.e[i][0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.stencil.e[i][0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.stencil.e[i][1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.stencil.e[i][1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
                    if c[p] == 0 and self.stencil.e[i][2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.stencil.e[i][2] == 1:  # searching border on right
                        border[2] = 1

                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not self.obstacle_mask[a[p] + self.stencil.e[i][0] - border[0] * nx,
                        b[p] + self.stencil.e[i][1] - border[1] * ny,
                        c[p] + self.stencil.e[i][2] - border[2] * nz]:

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.stencil.e[i][0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.stencil.e[i][1] - border[1] * ny  # fluid node y-coordinate
                            # Z-coordinate not needed for cylinder !

                            cx = self.stencil.e[self.stencil.opposite[i]][0]  # link-direction x to solid node
                            cy = self.stencil.e[self.stencil.opposite[i]][1]  # link-direction y to solid node
                            # Z-coordinate not needed for cylinder !

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
                                    d_lt.append(d1)
                                    f_index_lt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny,
                                                            c[p] + self.stencil.e[i][2] - border[2] * nz])
                                else:  # d>0.5
                                    d_gt.append(d1)
                                    f_index_gt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny,
                                                            c[p] + self.stencil.e[i][2] - border[2] * nz])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    d_lt.append(d2)
                                    f_index_lt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny,
                                                            c[p] + self.stencil.e[i][2] - border[2] * nz])
                                else:  # d>0.5
                                    d_gt.append(d2)
                                    f_index_gt.append([self.stencil.opposite[i],
                                                            a[p] + self.stencil.e[i][0] - border[0] * nx,
                                                            b[p] + self.stencil.e[i][1] - border[1] * ny,
                                                            c[p] + self.stencil.e[i][2] - border[2] * nz])
                            else:  # neither d1 nor d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,z,ci", a[p],
                                      b[p], c[p], self.stencil.e[i][0], self.stencil.e[i][1],
                                      self.stencil.e[i][2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # output as np.array: f_index_lt, f_index_gt, d_lt, d_gt
        return [np.array(f_index_lt), np.array(f_index_gt), np.array(d_lt), np.array(d_gt)]


    @property
    def grid(self):
        # THIS IS NOT USED AT THE MOMENT. QUESTION: SHOULD THIS BE ONE- OR ZERO-BASED? Indexing or "node-number"?
        xyz = tuple(self.units.convert_length_to_pu(np.linspace(0, n, n)) for n in self.resolution)  # tuple of lists of x,y,(z)-values/indices
        return np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (und z-)values/indices

    @property
    def post_boundaries(self):

        # inlet ("left side", x[0],y[1:-1], z[:])
        inlet_boundary = EquilibriumBoundaryPU(flow=self, context=self.context,
            mask=self.in_mask,
            velocity=self.u_inlet)  # (is this still true??): works with a 1 x D vector or an ny x D vector thanks to einsum-magic in EquilibriumBoundaryPU

        lateral_boundary = None  # stays None if lateral walls are not specified... (NOT IMPLEMENTED YET, see below)
        # >>> lateral walls ("top and bottom walls", x[:], y[0,-1], z[:])
        #TODO: implement lateral walls (top, bottom) for cylinder in channel (with or without wall friction)
        #   - implementation depends on positional definition: mask or index-lists or solid_boundary_data object!
        #
        # if self.lateral_walls == 'bounceback':
        #     if self.bc_type == 'hwbb' or self.bc_type == 'HWBB':  # use halfway bounce back
        #         lateral_boundary = HalfwayBounceBackBoundary(self.wall_mask, self.units.lattice)
        #     else:  # else use fullway bounce back
        #         lateral_boundary = FullwayBounceBackBoundary(self.wall_mask, self.units.lattice)
        # elif self.lateral_walls == 'slip' or self.bc_type == 'SLIP':  # use slip-walöl (symmetry boundary)
        #     lateral_boundary = SlipBoundary(self.wall_mask, self.units.lattice, 1)  # slip on x(z)-plane
        # <<<

        # outlet ("right side", x[-1],y[:], (z[:]))
        if self.stencil.d == 2:
            outlet_boundary = EquilibriumOutletP(direction=[1, 0], flow=self)  # outlet in positive x-direction
        else:  # self.stencil.d == 3:
            outlet_boundary = EquilibriumOutletP(direction=[1, 0, 0], flow=self)  # outlet in positive x-direction

        # create and return boundary-list
        if lateral_boundary is None:  # if lateral boundary is periodic...don't include the lateral_boundary object in the boundaries-list
            return [
                inlet_boundary,
                outlet_boundary,
                #BounceBackBoundary(self.context.convert_to_tensor(self.obstacle_mask,dtype=bool))
            ]
        else:
            return [
                inlet_boundary,
                outlet_boundary,
                lateral_boundary,
            ]

    @property
    def post_streaming_boundaries(self):
        # obstacle (for example: obstacle "cylinder" with radius centered at position x_pos, y_pos) -> to be set via obstacle_mask.setter
        obstacle_boundary = None
        # (!) the obstacle_boundary should always be the last boundary in the list of boundaries to correctly calculate forces on the obstacle

        solid_boundary_data = SolidBoundaryData()
        solid_boundary_data.solid_mask = self.obstacle_mask

        # load index-lists for circular cylinder into SBD object [f_index_lt, f_index_gt, d_lt, d_gt]:
        solid_boundary_data.f_index_lt, solid_boundary_data.f_index_gt, solid_boundary_data.d_lt, solid_boundary_data.d_gt = self.make_ibb_index_lists(
            x_center=self.x_pos_lu-1, y_center=self.y_pos_lu-1, radius=self.radius_lu)

        print("CYLINDER: the bc_type was given as:", self.bc_type)
        if self.bc_type.casefold() == 'fwbb' or self.bc_type == 'FWBB':
            obstacle_boundary = FullwayBounceBackBoundary(self.context, self,
                                                          self.obstacle_mask, periodicity = self.periodicity, calc_force=self.calc_force_coefficients)
        elif self.bc_type.casefold() == 'hwbb' or self.bc_type == 'HWBB':
            obstacle_boundary = HalfwayBounceBackBoundary(self.context, self,
                                                          solid_boundary_data, periodicity = self.periodicity, calc_force=self.calc_force_coefficients)
        elif self.bc_type.casefold() == 'ibb1' or self.bc_type == 'IBB1':
            obstacle_boundary = LinearInterpolatedBounceBackBoundary(self.context, self, solid_boundary_data, calc_force=self.calc_force_coefficients)
        else:  # use basic mask Bounce Back
            print("OBSTACLE CYLINDER - WARNING: bc_type can't be interpreted... - will fall back to using BounceBackBoundary")
            obstacle_boundary = BounceBackBoundary(self.context.convert_to_tensor(self.obstacle_mask))

        return [obstacle_boundary]

    def _unit_vector(self, i=0):
        return np.eye(self.stencil.d)[i]