"""
Boundary Conditions.

Boundary conditions take a mask (a boolean numpy array) and specifies the grid points on which the boundary
condition operates.
"""

import torch
import numpy as np


class BounceBackBoundary:
    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask.byte(), f[self.lattice.stencil.opposite], f)
        return f


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

class AntiBounceBackOutlet2D:
    # Seite 195 im Buch

    def __init__(self, grid, lattice, direction):
        self.mask = torch.zeros(lattice.convert_to_tensor(grid).shape[1:], device=lattice.device, dtype=torch.bool)
        #WIP here
        self.mask[-1, :] = True
        self.lattice = lattice
        #self.direction = [1, 5, 8]
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction)==1), axis=0)
        self.index2 = []
        #self.index = np.where(direction == 0, [slice(None)], [slice((direction + 1)*-0.5)])
        for i in direction:
            if i == 0:
                self.index2.append(slice(None))
            if i == 1:
                self.index2.append(-1)
            if i == -1:
                self.index2.append(0)
        # indices auf den stencils, wo z.B. x == 1 ist, oder y == 1 ect.)
        print("test")

    def __call__(self, f):

        u = self.lattice.u(f)
        #WIP here
        u_w = u[[slice(None)]+self.index2] + 0.5 * (u[:, -1, :] - u[:, -2, :])
        #f_bounced = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        for i in self.velocities:
            # formula according to book
            #f[i, -1, :] = - f_bounced[i, -1, :] + 2 * self.lattice.stencil.w[i] * self.lattice.rho(f)[0, self.mask] * \
            #            (1 + torch.matmul(torch.tensor((self.lattice.stencil.e[i]), device=f.device, dtype=f.dtype), u_w)**2 / (2 * self.lattice.stencil.cs**4) - torch.norm(u_w, dim=0)**2 / (2 * self.lattice.stencil.cs**2))
            # changed brackets ect. so it runs a bit faster:
            f[self.lattice.stencil.opposite[i], -1, :] = - f[i, -1, :] + self.lattice.stencil.w[i] * self.lattice.rho(f)[0, self.mask] * \
                        (2 + torch.matmul(torch.tensor((self.lattice.stencil.e[i]), device=f.device, dtype=f.dtype), u_w) ** 2 / self.lattice.stencil.cs ** 4 - (torch.norm(u_w, dim=0) / self.lattice.stencil.cs)**2)
        return f

class AntiBounceBackOutlet3D:
    # Seite 195 im Buch

    def __init__(self, grid, lattice, direction):
        self.mask = torch.zeros(lattice.convert_to_tensor(grid).shape[1:], device=lattice.device, dtype=torch.bool)
        self.mask[-1, :, :] = True
        self.lattice = lattice
        #self.direction = [1, 11, 13, 15, 17, 19, 21, 23, 25]
        self.direction = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) == 1), axis=0)
        print("test")

    def __call__(self, f):

        u = self.lattice.u(f)
        u_w = u[:, -1, :, :] + 0.5 * (u[:, -1, :, :] - u[:, -2, :, :])
        #f_bounced = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        for i in self.direction:
            #f[i, -1, :, :] = - f_bounced[i, -1, :, :] + 2 * self.lattice.stencil.w[i] * self.lattice.rho(f)[0, self.mask].reshape(f[0].shape[1], f[0].shape[2]) * \
            #              (1 + (torch.einsum('c, cyz -> yz', torch.tensor(self.lattice.stencil.e[i], device=f.device, dtype=f.dtype), u_w) ** 2 / (2 * self.lattice.stencil.cs ** 4) - torch.norm(u_w, dim=0) ** 2 / (2 * self.lattice.stencil.cs ** 2)))
            f[self.lattice.stencil.opposite[i], -1, :, :] = - f[i, -1, :, :] + self.lattice.stencil.w[i] * self.lattice.rho(f)[0, self.mask].reshape(f[0].shape[1], f[0].shape[2]) * \
                          (2 + torch.einsum('c, cyz -> yz', torch.tensor(self.lattice.stencil.e[i], device=f.device, dtype=f.dtype), u_w) ** 2 / self.lattice.stencil.cs ** 4 - (torch.norm(u_w, dim=0) / self.lattice.stencil.cs) ** 2)

        return f


