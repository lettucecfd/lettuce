import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary#, ZeroGradientOutletRight, ZeroGradientOutletTop, \
    #ZeroGradientOutletBottom


class Obstacle2D(object):

    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = None

    def initizialize_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array(
            [0 * x[0] + self.units.convert_velocity_to_lu(self.units.characteristic_velocity_pu), x[1] * 0],#.05],
            dtype=float)

    def getMaxU(self, f, lattice):
        u0 = (lattice.u(f)[0])
        u1 = (lattice.u(f)[1])
        return torch.max(torch.sqrt(u0*u0+u1*u1))

    def getSheddingFrequency(self, sample):
        #sample = np.asarray(sample)
        x = np.arange(sample.size)
        fourier = np.fft.fft(sample)
        return x[(fourier == np.min(fourier[2:int(sample.size/2)]))]/ sample.size

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution_y, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    #@property
    #def grid(self):
    #    x = np.linspace(0, self.resolution_x/self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
    #    y = np.linspace(0, self.resolution_y/self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
    #    return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, np.array(
            [self.units.characteristic_velocity_pu, self.units.characteristic_velocity_pu * 0.0])),
                #ZeroGradientOutletRight(np.abs(y-1) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
                #ZeroGradientOutletBottom(np.abs(x-1) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
                #ZeroGradientOutletTop(np.abs(x) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
                AntiBounceBackOutletBack2D(self.grid, self.units.lattice, 5), #inputs don't matter yet
                BounceBackBoundary(self.mask, self.units.lattice)]

#----------------------------------------------3D-----------------------------------------------------------------------


class Obstacle3D(object):

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = None

    def initizialize_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([0 * x[0]], dtype=float), np.array(
            [0 * x[0] + self.units.convert_velocity_to_lu(1.0), x[1] * 0, x[2] * 0],#.05],
            dtype=float)

    def getMaxU(self, f, lattice):
        u0 = (lattice.u(f)[0])
        u1 = (lattice.u(f)[1])
        u2 = (lattice.u(f)[2])
        return torch.max(torch.sqrt(u0 * u0 + u1 * u1 + u2 * u2))


    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution_y, endpoint=False)
        z = np.linspace(0, 1, num=self.resolution_z, endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, np.array(
            [self.units.characteristic_velocity_pu, self.units.characteristic_velocity_pu * 0.0, self.units.characteristic_velocity_pu * 0.0])),
                #ZeroGradientOutletRight(np.abs(y - 1) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
                # ZeroGradientOutletBottom(np.abs(x-1) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
                # ZeroGradientOutletTop(np.abs(x) < 1e-3, self.units.lattice, direction=[1.0, 0.0]),
                AntiBounceBackOutletBack3D(self.grid, self.units.lattice, 5),  # inputs don't matter yet
                BounceBackBoundary(self.mask, self.units.lattice)]

class ZeroGradientOutletTop:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[2, :, 0] = f[2, :, 1]
        f[5, :, 0] = f[5, :, 1]
        f[6, :, 0] = f[6, :, 1]
        return f

class AntiBounceBackOutletBack2D:
    # Seite 195 im Buch

    def __init__(self, grid, lattice, direction):
        self.mask = torch.zeros(lattice.convert_to_tensor(grid).shape[1:], device=lattice.device, dtype=torch.bool)
        self.mask[-1, :] = True
        self.lattice = lattice
        self.direction = [1, 5, 8]

    def __call__(self, f):

        u = self.lattice.u(f)
        u_w = u[:, -1, :] + 0.5 * (u[:, -1, :] - u[:, -2, :])
        #f_bounced = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        for i in self.direction:
            # formula according to book
            #f[i, -1, :] = - f_bounced[i, -1, :] + 2 * self.lattice.stencil.w[i] * self.lattice.rho(f)[0, self.mask] * \
            #            (1 + torch.matmul(torch.tensor((self.lattice.stencil.e[i]), device=f.device, dtype=f.dtype), u_w)**2 / (2 * self.lattice.stencil.cs**4) - torch.norm(u_w, dim=0)**2 / (2 * self.lattice.stencil.cs**2))
            # changed brackets ect. so it runs faster:
            f[self.lattice.stencil.opposite[i], -1, :] = - f[i, -1, :] + self.lattice.stencil.w[i] * self.lattice.rho(f)[0, self.mask] * \
                        (2 + torch.matmul(torch.tensor((self.lattice.stencil.e[i]), device=f.device, dtype=f.dtype), u_w) ** 2 / self.lattice.stencil.cs ** 4 - (torch.norm(u_w, dim=0) / self.lattice.stencil.cs)**2)
        return f

class AntiBounceBackOutletBack3D:
    # Seite 195 im Buch

    def __init__(self, grid, lattice, direction):
        self.mask = torch.zeros(lattice.convert_to_tensor(grid).shape[1:], device=lattice.device, dtype=torch.bool)
        self.mask[-1, :, :] = True
        self.lattice = lattice
        self.direction = [1, 11, 13, 15, 17, 19, 21, 23, 25]

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

class ZeroGradientOutletBottom:
    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.direction = direction

    def __call__(self, f):
        f[4, :, -1] = f[4, :, -2]
        f[7, :, -1] = f[7, :, -2]
        f[8, :, -1] = f[8, :, -2]
        return f


class uSampleReporter:
    def __init__(self, lattice, flow, interval=50):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.out = []

    def __call__(self, i, t, f):
        if t % self.interval == 0:
            if t > 1000:
                x_dim = f.shape[1]
                y_dim = f.shape[2]
                u = self.lattice.u(f)[:, int(x_dim / 3), int(y_dim * 0.6)]
                u_0 = u[0]
                self.out.append(u_0.cpu().numpy())
