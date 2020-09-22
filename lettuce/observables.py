"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""


import torch
import numpy as np
from lettuce.util import torch_gradient


__all__ = ["MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum"]


class MaximumVelocity:
    """Maximum velocitiy"""
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.flow.units.convert_velocity_to_pu(torch.norm(u, dim=0).max())


class IncompressibleKineticEnergy:
    """Total kinetic energy of an incompressible flow."""
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)))
        kinE *= dx ** self.lattice.D
        return kinE


class Enstrophy:
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

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
        return vorticity * dx**self.lattice.D


class EnergySpectrum:
    """The kinetic energy spectrum"""
    def __init__(self, lattice, flow):
        self.lattice =lattice
        self.flow = flow
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies))
        wavenorms = torch.norm(wavenumbers, dim=0)
        self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2 if self.lattice.D == 3 else self.dimensions[0] / self.dx
        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
            (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype) - 0.5) &
            (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype) + 0.5)
        )

    def __call__(self, f):
        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        zeros = torch.zeros(self.dimensions, dtype=self.lattice.dtype, device=self.lattice.device)[..., None]
        uh = (torch.stack([
            torch.fft(torch.cat((u[i][..., None], zeros), self.lattice.D),
                      signal_ndim=self.lattice.D) for i in range(self.lattice.D)]) / self.norm)
        ekin = torch.sum(0.5 * (uh[...,0]**2 + uh[...,1]**2), dim=0)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.lattice.dtype)
        ek = ek.sum(torch.arange(self.lattice.D).tolist())
        return ek


