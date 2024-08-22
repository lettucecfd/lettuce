import sys
import torch
import numpy as np

from ... import Reporter, Flow
from ...util import torch_gradient
from packaging import version

__all__ = ['Observable', 'ObservableReporter', 'MaximumVelocity',
           'IncompressibleKineticEnergy', 'Enstrophy', 'EnergySpectrum',
           'Mass']


class Observable:
    def __init__(self, flow: Flow):
        self.context = flow.context
        self.flow = flow

    def __call__(self, f):
        raise NotImplementedError


class MaximumVelocity(Observable):
    """Maximum velocitiy"""

    def __call__(self, f):
        return torch.norm(self.flow.u_pu, dim=0).max()


class IncompressibleKineticEnergy(Observable):
    """Total kinetic energy of an incompressible flow."""

    def __call__(self, f):
        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(
            torch.sum(self.flow.incompressible_energy()))
        kinE *= dx ** self.flow.stencil.d
        return kinE


class Enstrophy(Observable):
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """

    def __call__(self, f):
        u0 = self.flow.units.convert_velocity_to_pu(self.flow.u()[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.flow.u()[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = torch.sum((grad_u0[1] - grad_u1[0])
                              * (grad_u0[1] - grad_u1[0]))
        if self.flow.stencil.d == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.flow.u()[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity * dx ** self.flow.stencil.d


class EnergySpectrum(Observable):
    """The kinetic energy spectrum"""

    def __init__(self, flow: Flow):
        super(EnergySpectrum, self).__init__(flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.context.convert_to_tensor(
            np.fft.fftfreq(dim, d=1 / dim)
        ) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies, indexing='ij'))
        wavenorms = torch.norm(wavenumbers, dim=0)

        if self.flow.stencil.d == 3:
            self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2
        else:
            self.norm = self.dimensions[0] / self.dx

        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
                (wavenorms[..., None] > self.wavenumbers.to(
                    dtype=self.context.dtype, device=self.context.device)
                 - 0.5) &
                (wavenorms[..., None] <= self.wavenumbers.to(
                    dtype=self.context.dtype, device=self.context.device)
                 + 0.5)
        )

    def __call__(self, f):
        u = self.flow.u()
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
        ekin = self._ekin_spectrum(u)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.context.dtype)
        ek = ek.sum(torch.arange(self.flow.stencil.d).tolist())
        return ek

    def _ekin_spectrum(self, u):
        """distinguish between different torch versions"""
        torch_ge_18 = (version.parse(torch.__version__) >= version.parse(
            "1.8.0"))
        if torch_ge_18:
            return self._ekin_spectrum_torch_ge_18(u)
        else:
            return self._ekin_spectrum_torch_lt_18(u)

    def _ekin_spectrum_torch_lt_18(self, u):
        zeros = torch.zeros(self.dimensions, dtype=self.context.dtype,
                            device=self.context.device)[..., None]
        uh = (torch.stack(
            [torch.fft(torch.cat((u[i][..., None], zeros),
                                 self.flow.stencil.d),
                       signal_ndim=self.flow.stencil.d)
             for i in range(self.flow.stencil.d)
             ]) / self.norm)
        ekin = torch.sum(0.5 * (uh[..., 0] ** 2 + uh[..., 1] ** 2), dim=0)
        return ekin

    def _ekin_spectrum_torch_ge_18(self, u):
        uh = (torch.stack([
            torch.fft.fftn(u[i], dim=tuple(torch.arange(self.flow.stencil.d)))
            for i in range(self.flow.stencil.d)
        ]) / self.norm)
        ekin = torch.sum(0.5 * (uh.imag ** 2 + uh.real ** 2), dim=0)
        return ekin


class Mass(Observable):
    """Total mass in lattice units.

    Parameters
    ----------
    no_mass_mask : torch.Tensor
        Boolean mask that defines grid points
        which do not count into the total mass (e.g. bounce-back boundary).
    """

    def __init__(self, flow: Flow, no_mass_mask=None):
        super(Mass, self).__init__(flow)
        self.mask = no_mass_mask

    def __call__(self, f):
        mass = f[..., 1:-1, 1:-1].sum()
        if self.mask is not None:
            mass -= (f * self.mask.to(dtype=torch.float)).sum()
        return mass


class ObservableReporter(Reporter):
    """A reporter that prints an observable every few iterations.

    Examples
    --------
    Create an Enstrophy reporter.

    >>> from lettuce import TaylorGreenVortex3D, Enstrophy, D3Q27, Context
    >>> context = Context(device=torch.device("cpu"))
    >>> flow = TaylorGreenVortex(context, 50, 300, 0.1, D3Q27())
    >>> simulation = ...
    >>> enstrophy = Enstrophy(flow)
    >>> reporter = ObservableReporter(enstrophy, interval=10)
    >>> simulation.reporter.append(reporter)
    """

    def __init__(self, observable, interval=1, out=sys.stdout):
        super().__init__(interval)
        self.observable = observable
        self.out = [] if out is None else out
        self._parameter_name = observable.__class__.__name__
        print('steps    ', 'time    ', self._parameter_name)

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i % self.interval == 0:
            observed = self.observable.context.convert_to_ndarray(
                self.observable(simulation.flow.f))
            assert len(observed.shape) < 2
            if len(observed.shape) == 0:
                observed = [observed.item()]
            else:
                observed = observed.tolist()
            entry = ([simulation.flow.i,
                      simulation.units.convert_time_to_pu(simulation.flow.i)]
                     + observed)
            if isinstance(self.out, list):
                self.out.append(entry)
            else:
                print(*entry, file=self.out)
