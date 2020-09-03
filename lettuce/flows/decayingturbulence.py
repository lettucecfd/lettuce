"""
DecayingTurbulence vortex in 2D.
"""

import numpy as np
from lettuce.unit import UnitConversion


class DecayingTurbulence2D:
    """Decaying isotropic turbulence in 2D"""
    def __init__(self, resolution, reynolds_number, mach_number, lattice, k0=20, ic_energy=0.5):
        self.k0 = k0
        self.ic_energy = ic_energy
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2*np.pi,
            characteristic_velocity_pu=None
        )

    def analytic_solution(self, x, t=0):
        return

    def initial_solution(self, x):
        """Return initial solution. Note: this function sets the characteristic velocity in phyiscal units."""
        dx = self.units.convert_length_to_pu(1.0)

        ### Generate wavenumber vector
        kx = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        ky = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        kx, ky = np.meshgrid(kx, ky)
        kk = np.sqrt(kx ** 2 + ky ** 2)
        kk[0][0] = 1e-16

        ### Generate spectrum
        ek = (kk) ** 4 * np.exp(-2 * (kk / self.k0) ** 2)
        ek[0][0] = 0
        ek /= np.sum(ek)
        ek *= self.ic_energy

        # Forward transform random fields
        u0 = np.random.randn(self.resolution,self.resolution) + 0j
        u1 = np.random.randn(self.resolution,self.resolution) + 0j

        u0 = np.fft.fftn(u0, axes=(0, 1))
        u1 = np.fft.fftn(u1, axes=(0, 1))
        # real parts
        u0_real = u0.real
        u1_real = u1.real
        # imaginary parts
        u0_imag = u0.imag
        u1_imag = u1.imag

        # no mean value
        u0_real[0][0] = 0
        u0_imag[0][0] = 0
        u1_real[0][0] = 0
        u1_imag[0][0] = 0

        # scale with target energy at kk and divide by local energy at kk to force the spectrum
        u0_real_h = np.sqrt(2 / self.units.lattice.D * ek / (u0_imag ** 2 + u0_real ** 2 + 1.e-15)) * u0_real
        u0_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (u0_imag ** 2 + u0_real ** 2 + 1.e-15)) * u0_imag
        u1_real_h = np.sqrt(2 / self.units.lattice.D * ek / (u1_imag ** 2 + u1_real ** 2 + 1.e-15)) * u1_real
        u1_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (u1_imag ** 2 + u1_real ** 2 + 1.e-15)) * u1_imag
        u0_real_h[0][0] = 0
        u1_real_h[0][0] = 0
        u0_imag_h[0][0] = 0
        u1_imag_h[0][0] = 0

        ### Remove divergence
        # modified wave number sin(k*dx) is used, as the gradient below uses second order cental differences
        # Modify if other schemes are used or use kx, ky if you don't know the modified wavenumber !!!
        kx_modified = np.sin(kx * dx) / dx
        ky_modified = np.sin(ky * dx) / dx
        kk_modified = np.sqrt(kx_modified ** 2 + ky_modified ** 2 + 1e-16)

        divergence_real = (kx_modified * u0_real_h + ky_modified * u1_real_h)
        divergence_imag = (kx_modified * u0_imag_h + ky_modified * u1_imag_h)

        u0_real = u0_real_h - divergence_real * kx_modified / kk_modified ** 2
        u0_imag = u0_imag_h - divergence_imag * kx_modified / kk_modified ** 2
        u1_real = u1_real_h - divergence_real * ky_modified / kk_modified ** 2
        u1_imag = u1_imag_h - divergence_imag * ky_modified / kk_modified ** 2

        u0_real[0][0] = 0
        u0_imag[0][0] = 0
        u1_real[0][0] = 0
        u1_imag[0][0] = 0

        ### Scale velocity field to achieve the desired inicial energy
        e_kin = np.sum((u0_real ** 2 + u0_imag ** 2 + u1_real ** 2 + u1_imag ** 2)) * .5
        factor = np.sqrt(self.ic_energy / e_kin)
        u0_real *= factor
        u0_imag *= factor
        u1_real *= factor
        u1_imag *= factor

        ### Backtransformation to physical space
        norm = self.resolution / dx
        # backtransformation of u0
        u0 = u0_real + u0_imag * 1.0j
        u0f = np.fft.ifftn(u0, axes=(0, 1)) * norm
        u = u0f.real[None,...]

        # backtransformation of u1
        u1 = u1_real + u1_imag * 1.0j
        u1f = np.fft.ifftn(u1, axes=(0, 1)) * norm
        u = np.append(u, u1f.real[None, ...], axis=0)

        p = (u[0]*0)[None,...]

        umax = np.linalg.norm(u, axis=0).max()
        self.units.characteristic_velocity_pu = umax

        return p, u

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y)

    @property
    def boundaries(self):
        return []


class DecayingTurbulence3D:
    """Decaying isotropic turbulence in 3D"""
    def __init__(self, resolution, reynolds_number, mach_number, lattice, k0=20, ic_energy=0.5):
        self.k0 = k0
        self.ic_energy = ic_energy
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2*np.pi,
            characteristic_velocity_pu=None
        )

    def analytic_solution(self, x, t=0):
        return

    def initial_solution(self, x):
        """Return initial solution. Note: this function sets the characteristic velocity in phyiscal units."""
        dx = self.units.convert_length_to_pu(1.0)

        ### Generate wavenumber vector
        kx = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        ky = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        kz = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        kx, ky, kz = np.meshgrid(kx, ky, kz)
        kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
        kk[0][0][0] = 1e-16

        ### Generate spectrum
        ek = (kk) ** 4 * np.exp(-2 * (kk / self.k0) ** 2)
        ek[0][0][0] = 0
        ek /= np.sum(ek)
        ek *= self.ic_energy

        # Forward transform random fields
        u0 = np.random.randn(self.resolution,self.resolution,self.resolution) + 0j
        u1 = np.random.randn(self.resolution,self.resolution,self.resolution) + 0j
        u2 = np.random.randn(self.resolution,self.resolution,self.resolution) + 0j

        u0 = np.fft.fftn(u0, axes=(0, 1, 2))
        u1 = np.fft.fftn(u1, axes=(0, 1, 2))
        u2 = np.fft.fftn(u2, axes=(0, 1, 2))

        # real parts
        u0_real = u0.real
        u1_real = u1.real
        u2_real = u2.real
        # imaginary parts
        u0_imag = u0.imag
        u1_imag = u1.imag
        u2_imag = u2.imag

        # no mean value
        u0_real[0][0][0] = 0
        u0_imag[0][0][0] = 0
        u1_real[0][0][0] = 0
        u1_imag[0][0][0] = 0
        u2_real[0][0][0] = 0
        u2_imag[0][0][0] = 0

        # scale with target energy at kk and divide by local energy at kk to force the spectrum
        u0_real_h = np.sqrt(2 / self.units.lattice.D * ek / (u0_imag ** 2 + u0_real ** 2 + 1.e-15)) * u0_real
        u0_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (u0_imag ** 2 + u0_real ** 2 + 1.e-15)) * u0_imag
        u1_real_h = np.sqrt(2 / self.units.lattice.D * ek / (u1_imag ** 2 + u1_real ** 2 + 1.e-15)) * u1_real
        u1_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (u1_imag ** 2 + u1_real ** 2 + 1.e-15)) * u1_imag
        u2_real_h = np.sqrt(2 / self.units.lattice.D * ek / (u2_imag ** 2 + u2_real ** 2 + 1.e-15)) * u2_real
        u2_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (u2_imag ** 2 + u2_real ** 2 + 1.e-15)) * u2_imag
        u0_real_h[0][0][0] = 0
        u1_real_h[0][0][0] = 0
        u2_real_h[0][0][0] = 0
        u0_imag_h[0][0][0] = 0
        u1_imag_h[0][0][0] = 0
        u2_imag_h[0][0][0] = 0

        ### Remove divergence
        # modified wave number sin(k*dx) is used, as the gradient below uses second order cental differences
        # Modify if other schemes are used or use kx, ky if you don't know the modified wavenumber !!!
        kx_modified = np.sin(kx * dx) / dx
        ky_modified = np.sin(ky * dx) / dx
        kz_modified = np.sin(kz * dx) / dx
        kk_modified = np.sqrt(kx_modified ** 2 + ky_modified ** 2 + kz_modified ** 2 + 1e-16)

        divergence_real = (kx_modified * u0_real_h + ky_modified * u1_real_h + kz_modified * u2_real_h)
        divergence_imag = (kx_modified * u0_imag_h + ky_modified * u1_imag_h + kz_modified * u2_imag_h)

        u0_real = u0_real_h - divergence_real * kx_modified / kk_modified ** 2
        u0_imag = u0_imag_h - divergence_imag * kx_modified / kk_modified ** 2
        u1_real = u1_real_h - divergence_real * ky_modified / kk_modified ** 2
        u1_imag = u1_imag_h - divergence_imag * ky_modified / kk_modified ** 2
        u2_real = u2_real_h - divergence_real * kz_modified / kk_modified ** 2
        u2_imag = u2_imag_h - divergence_imag * kz_modified / kk_modified ** 2

        u0_real[0][0][0] = 0
        u0_imag[0][0][0] = 0
        u1_real[0][0][0] = 0
        u1_imag[0][0][0] = 0
        u2_real[0][0][0] = 0
        u2_imag[0][0][0] = 0

        ### Scale velocity field to achieve the desired inicial energy
        e_kin = np.sum((u0_real ** 2 + u0_imag ** 2 + u1_real ** 2 + u1_imag ** 2 + u2_real ** 2 + u2_imag ** 2)) * .5
        factor = np.sqrt(self.ic_energy / e_kin)
        u0_real *= factor
        u0_imag *= factor
        u1_real *= factor
        u1_imag *= factor
        u2_real *= factor
        u2_imag *= factor

        ### Backtransformation to physical space
        norm = self.resolution * dx ** (-2) * np.sqrt(self.units.characteristic_length_pu)

        # backtransformation of u0
        u0 = u0_real + u0_imag * 1.0j
        u0f = np.fft.ifftn(u0, axes=(0, 1, 2)) * norm
        u = u0f.real[None,...]

        # backtransformation of u1
        u1 = u1_real + u1_imag * 1.0j
        u1f = np.fft.ifftn(u1, axes=(0, 1, 2)) * norm
        u = np.append(u, u1f.real[None, ...], axis=0)

        # backtransformation of u2
        u2 = u2_real + u2_imag * 1.0j
        u2f = np.fft.ifftn(u2, axes=(0, 1, 2)) * norm
        u = np.append(u, u2f.real[None, ...], axis=0)

        p = (u[0]*0)[None,...]

        umax = np.linalg.norm(u, axis=0).max()
        self.units.characteristic_velocity_pu = umax

        return p, u

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        z = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, z)

    @property
    def boundaries(self):
        return []
