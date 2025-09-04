"""
DecayingTurbulence vortex in 2D and 3D. Dimension is set by the stencil.
Special Inputs & standard value: wavenumber_energy-peak = 20,
initial_energy = 0.5

Additional attributes / properties
__________
energy_spectrum: returns a pair [spectrum, wavenumbers]
"""
from typing import Union, List, Optional

import numpy as np
import torch

from ... import UnitConversion
from .._stencil import D1Q3, D2Q9, D3Q19
from . import ExtFlow


__all__ = ['DecayingTurbulence']


class DecayingTurbulence(ExtFlow):

    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number, k0=20, ic_energy=0.5,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 initialize_pressure: bool = True,
                 initialize_fneq: bool = True,
                 randseed: Optional[int] = None):
        self.initialize_pressure = initialize_pressure
        self.initialize_fneq = initialize_fneq
        self.randseed = randseed
        self.k0 = k0
        self.ic_energy = ic_energy
        self.wavenumbers = []
        self.spectrum = []
        default_stencils = [D1Q3(), D2Q9(), D3Q19()]
        stencil = stencil or default_stencils[len(resolution) - 1]
        stencil = stencil() if callable(stencil) else stencil
        if stencil.d != 2:
            self.initialize_pressure = False
        super().__init__(context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * stencil.d
        else:
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution
                   ) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=resolution[0],
            characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=None
        )

    def analytic_solution(self, x, t=0):
        return

    def _generate_wavenumbers(self):
        self.dimensions = self.grid[0].shape
        frequencies = [np.fft.fftfreq(dim, d=1 / dim)
                       for dim in self.dimensions]
        wavenumber = np.meshgrid(*frequencies)
        wavenorms = np.linalg.norm(wavenumber, axis=0)
        self.wavenumbers = np.arange(int(np.max(wavenorms)))
        wavemask = ((wavenorms[..., None] > self.wavenumbers - 0.5)
                    & (wavenorms[..., None] <= self.wavenumbers + 0.5))
        return wavenorms, wavenumber, wavemask

    def _generate_spectrum(self):
        wavenorms, wavenumber, wavemask = self._generate_wavenumbers()
        ek = wavenorms ** 4 * np.exp(-2 * (wavenorms / self.k0) ** 2)
        ek /= np.sum(ek)
        ek *= self.ic_energy
        self.spectrum = ek[..., None] * wavemask
        self.spectrum = np.sum(self.spectrum,
                               axis=tuple((np.arange(self.stencil.d))))
        return ek, wavenumber

    def _generate_initial_velocity(self, ek, wavenumber):
        dx = self.units.convert_length_to_pu(1.0)
        np.random.seed(self.randseed)
        u = np.random.random(np.array(wavenumber).shape) * 2 * np.pi + 0j
        u = [np.fft.fftn(u[dim], axes=tuple((np.arange(self.stencil.d))))
             for
             dim in range(self.stencil.d)]

        u_real = [u[dim].real for dim in range(self.stencil.d)]
        u_imag = [u[dim].imag for dim in range(self.stencil.d)]
        for dim in range(self.stencil.d):
            u_real[dim].ravel()[0] = 0
            u_imag[dim].ravel()[0] = 0

        u_real_h = [np.sqrt(2 / self.stencil.d * ek
                            / (u_imag[dim] ** 2 + u_real[dim] ** 2 + 1.e-15))
                    * u_real[dim] for dim in range(self.stencil.d)]
        u_imag_h = [np.sqrt(2 / self.stencil.d * ek
                            / (u_imag[dim] ** 2 + u_real[dim] ** 2 + 1.e-15))
                    * u_imag[dim] for dim in range(self.stencil.d)]
        for dim in range(self.stencil.d):
            u_real_h[dim].ravel()[0] = 0
            u_imag_h[dim].ravel()[0] = 0

        """ Remove divergence
        # modified wave number sin(k*dx) is used, as the gradient below uses
        # second order cental differences
        # Modify if other schemes are used or use kx, ky if you don't know
        # the modified wavenumber !!!
        """
        wavenumber_modified = [np.sin(wavenumber[dim] * dx) / dx
                               for dim in range(self.stencil.d)]
        wavenorm_modified = np.linalg.norm(wavenumber_modified, axis=0) + 1e-16

        divergence_real = np.zeros(self.dimensions)
        divergence_imag = np.zeros(self.dimensions)
        for dim in range(self.stencil.d):
            divergence_real += wavenumber_modified[dim] * u_real_h[dim]
            divergence_imag += wavenumber_modified[dim] * u_imag_h[dim]

        u_real = [u_real_h[dim] - divergence_real * wavenumber_modified[dim]
                  / wavenorm_modified ** 2 for dim in range(self.stencil.d)]
        u_imag = [u_imag_h[dim] - divergence_imag * wavenumber_modified[dim]
                  / wavenorm_modified ** 2 for dim in range(self.stencil.d)]
        for dim in range(self.stencil.d):
            u_real[dim].ravel()[0] = 0
            u_imag[dim].ravel()[0] = 0

        # Scale velocity field to achieve the desired inicial energy
        e_kin = [np.sum(u_real[dim] ** 2 + u_imag[dim] ** 2)
                 for dim in range(self.stencil.d)]
        e_kin = np.sum(e_kin) * .5

        factor = np.sqrt(self.ic_energy / e_kin)
        u_real = [u_real[dim] * factor for dim in range(self.stencil.d)]
        u_imag = [u_imag[dim] * factor for dim in range(self.stencil.d)]

        # Backtransformation to physical space
        norm = ((self.resolution[0] * dx ** (1 - self.stencil.d) * np.sqrt(
            self.units.characteristic_length_pu))
                if self.stencil.d == 3 else (self.resolution[0] / dx))

        u = np.asarray([
            (np.fft.ifftn(u_real[dim] + u_imag[dim] * 1.0j,
                          axes=tuple((np.arange(self.stencil.d)))) * norm).real
            for dim in range(self.stencil.d)])

        return u

    def _compute_initial_pressure(self):
        # TODO: use pressure_poisson (@12b02686) to approximate rho from
        #  u-field
        return np.zeros(self.dimensions)[None, ...]

    def initial_pu(self):
        """Return initial solution. Note: this function sets the
        characteristic velocity in phyiscal units."""
        ek, wavenumber = self._generate_spectrum()
        u = self._generate_initial_velocity(ek, wavenumber)
        self.units.characteristic_velocity_pu = np.array(u).max()
        p = self._compute_initial_pressure()
        self.units.characteristic_velocity_pu = np.linalg.norm(u, axis=0).max()
        return p, u

    @property
    def energy_spectrum(self):
        return self.spectrum, self.wavenumbers

    @property
    def grid(self) -> (torch.Tensor, ...):
        endpoints = [2 * torch.pi * (1 - 1 / n) for n in
                     self.resolution]  # like endpoint=False in np.linspace
        xyz = tuple(torch.linspace(0, endpoints[n],
                                   steps=self.resolution[n],
                                   device=self.context.device,
                                   dtype=self.context.dtype)
                    for n in range(self.stencil.d))
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def post_boundaries(self) -> List['Boundary']:
        return []
