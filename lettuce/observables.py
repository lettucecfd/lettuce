"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""

import torch
import numpy as np
from lettuce.util import torch_gradient
from packaging import version

__all__ = ["Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum",
           "IncompressibleKineticEnergyBd","Dissipation_sij","Dissipation_TGV"]


class Observable:
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        raise NotImplementedError


class MaximumVelocity(Observable):
    """Maximum velocitiy"""

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.flow.units.convert_velocity_to_pu(torch.norm(u, dim=0).max())


class IncompressibleKineticEnergy(Observable):
    """Total kinetic energy of an incompressible flow."""

    def __call__(self, f):

        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)))
        kinE *= dx ** self.lattice.D
        return kinE


class IncompressibleKineticEnergyBd(Observable):
    """Total kinetic energy of an incompressible flow."""

    def __call__(self, f):

        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)[1:-1,1:-1]))
        kinE *= dx ** self.lattice.D
        return kinE





class Enstrophy(Observable):
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """

    def __call__(self, f):
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity * dx ** self.lattice.D


class EnergySpectrum(Observable):
    """The kinetic energy spectrum"""

    def __init__(self, lattice, flow):
        super(EnergySpectrum, self).__init__(lattice, flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies))
        wavenorms = torch.norm(wavenumbers, dim=0)

        if self.lattice.D == 3:
            self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2
        else:
            self.norm = self.dimensions[0] / self.dx

        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
                (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
                (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
        ekin = self._ekin_spectrum(u)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.lattice.dtype)
        ek = ek.sum(torch.arange(self.lattice.D).tolist())
        return ek

    def _ekin_spectrum(self, u):
        """distinguish between different torch versions"""
        torch_ge_18 = (version.parse(torch.__version__) >= version.parse("1.8.0"))
        if torch_ge_18:
            return self._ekin_spectrum_torch_ge_18(u)
        else:
            return self._ekin_spectrum_torch_lt_18(u)

    def _ekin_spectrum_torch_lt_18(self, u):
        zeros = torch.zeros(self.dimensions, dtype=self.lattice.dtype, device=self.lattice.device)[..., None]
        uh = (torch.stack([
            torch.fft(torch.cat((u[i][..., None], zeros), self.lattice.D),
                      signal_ndim=self.lattice.D) for i in range(self.lattice.D)]) / self.norm)
        ekin = torch.sum(0.5 * (uh[..., 0] ** 2 + uh[..., 1] ** 2), dim=0)
        return ekin

    def _ekin_spectrum_torch_ge_18(self, u):
        uh = (torch.stack([
            torch.fft.fftn(u[i], dim=tuple(torch.arange(self.lattice.D))) for i in range(self.lattice.D)
        ]) / self.norm)
        ekin = torch.sum(0.5 * (uh.imag ** 2 + uh.real ** 2), dim=0)
        return ekin


class Mass(Observable):
    """Total mass in lattice units.

    Parameters
    ----------
    no_mass_mask : torch.Tensor
        Boolean mask that defines grid points
        which do not count into the total mass (e.g. bounce-back boundaries).
    """

    def __init__(self, lattice, flow, no_mass_mask=None):
        super(Mass, self).__init__(lattice, flow)
        self.mask = no_mass_mask

    def __call__(self, f):
        mass = f[..., 1:-1, 1:-1].sum()
        if self.mask is not None:
            mass -= (f * self.mask.to(dtype=torch.float)).sum()
        return mass

class Dissipation_sij(Observable):

    def __init__(self, lattice, flow, no_grad=True):
        super(Dissipation_sij, self).__init__(lattice, flow)

    def __call__(self, f):
        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        dx = self.flow.units.convert_length_to_pu(1.0)
        nu = self.flow.units.viscosity_pu

        u_ij = torch.stack([torch_gradient(u[i], dx=dx, order=6) for i in range(self.lattice.D)])
        s_ij = 0.5 * (u_ij + torch.transpose(u_ij, 0, 1))
        dissipation = 2 * nu * torch.mean((s_ij ** 2).sum(0).sum(0))

        #du_dx=torch.gradient(u[0], dx=dx, order=6)
        #du_dy=torch.gradient(u[1], dx=dx, order=6)
        #du_dz=torch.gradient(u[2], dx=dx, order=6)
        #dissipation= 2*nu*(du_dx**2+du_dy**2+du_dz**2)
        return dissipation

class Dissipation_TGV(Observable):

    def __init__(self, lattice, flow, no_grad=True):
        super(Dissipation_TGV, self).__init__(lattice, flow)

    def __call__(self, f):
        

        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        dx = self.flow.units.convert_length_to_pu(1.0)
        nu = self.flow.units.viscosity_pu
        nges=u.size()[1]
        x=torch.linspace(0,(nges-1)*dx,steps=nges)
        y = torch.linspace(0,(nges - 1) * dx,steps=nges)
        z = torch.linspace(0,(nges - 1) * dx,steps=nges)
        grid=torch.meshgrid(x, y, z)
        u=u.requires_grad_()
        dux=torch.stack([torch.gradient(u[i],spacing=dx,dim=0) for i in range(3)])
        duy = torch.stack([torch.gradient(u[i], spacing=dx,dim=1) for i in range(3)])
        duz = torch.stack([torch.gradient(u[i], spacing=dx,dim=2) for i in range(3)])

        u_ijx=torch.square(dux[0])+torch.square(dux[1])+torch.square(dux[2])
        u_ijy = torch.square(duy[0]) + torch.square(duy[1]) + torch.square(duy[2])
        u_ijz = torch.square(duz[0]) + torch.square(duz[1]) + torch.square(duz[2])

        dissipation=nu/(u_ijx.size())**3*torch.sum(u_ijx+u_ijy+u_ijz)


        #u_ij = torch.stack([torch_gradient(u[i], dx=dx, order=6) for i in range(self.lattice.D)])
        #u_ijx=torch.square(u_ij[0,0])+torch.square(u_ij[0,1])+torch.square(u_ij[0,2])
        #u_ijy=torch.square(u_ij[1,0])+torch.square(u_ij[1,1])+torch.square(u_ij[1,2])
        #u_ijz=torch.square(u_ij[2,0])+torch.square(u_ij[2,1])+torch.square(u_ij[2,2])
        #dissipation=nu/(u_ijx.size()[0])**3*torch.sum(u_ijx+u_ijy+u_ijz)

        #du_dx=torch.gradient(u[0], dx=dx, order=6)
        #du_dy=torch.gradient(u[1], dx=dx, order=6)
        #du_dz=torch.gradient(u[2], dx=dx, order=6)
        #dissipation= 2*nu*(du_dx**2+du_dy**2+du_dz**2)
        return dissipation