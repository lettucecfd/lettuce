"""
DecayingTurbulence vortex in 2D.
"""

import numpy as np
from lettuce.unit import UnitConversion


class DecayingTurbulence2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice, k0=20, ic_energy=0.5):
        self.k0 = k0
        self.ic_energy = ic_energy
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2*np.pi,
            characteristic_velocity_pu=1
        )



    def analytic_solution(self, x, t=0):
        return

    def initial_solution(self, x):
        dx = self.units.characteristic_length_pu / self.resolution

        ### Generate wavenumber vector
        kx = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        ky = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        KX, KY = np.meshgrid(kx, ky)
        KK = np.sqrt(KX ** 2 + KY ** 2)
        KK[0][0] = 1e-16

        ### Generate spectrum
        ek = (KK) ** 4 * np.exp(-2 * (KK / self.k0) ** 2)
        ek[0][0] = 0
        ek /= np.sum(ek)
        ek *= self.ic_energy

        # Forward transform random fields
        u = np.random.randn(self.resolution,self.resolution) + 0j
        v = np.random.randn(self.resolution,self.resolution) + 0j

        uc = np.fft.fftn(u, axes=(0, 1))
        vc = np.fft.fftn(v, axes=(0, 1))
        # real parts
        ucr = uc.real
        vcr = vc.real
        # imaginary parts
        uci = uc.imag
        vci = vc.imag

        # no mean value
        ucr[0][0] = 0
        uci[0][0] = 0
        vcr[0][0] = 0
        vci[0][0] = 0

        # scale with target energy at KK and divide by local energy at KK to force the spectrum
        u0_real_h = np.sqrt(2 / self.units.lattice.D * ek / (uci ** 2 + ucr ** 2 + 1.e-15)) * ucr
        u0_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (uci ** 2 + ucr ** 2 + 1.e-15)) * uci
        u1_real_h = np.sqrt(2 / self.units.lattice.D * ek / (vci ** 2 + vcr ** 2 + 1.e-15)) * vcr
        u1_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (vci ** 2 + vcr ** 2 + 1.e-15)) * vci
        u0_real_h[0][0] = 0
        u1_real_h[0][0] = 0
        u0_imag_h[0][0] = 0
        u1_imag_h[0][0] = 0

        ### Remove divergence
        # modified wave number sin(k*dx) is used, as the gradient below uses second order cental differences
        # Modify if other schemes are used or use KX, KY if you don't know the modified wavenumber !!!
        KXm = np.sin(KX * dx) / dx
        KYm = np.sin(KY * dx) / dx
        KKm = np.sqrt(KXm ** 2 + KYm ** 2 + 1e-16)

        divr = (KXm * u0_real_h + KYm * u1_real_h)
        divi = (KXm * u0_imag_h + KYm * u1_imag_h)

        ucr = u0_real_h - divr * KXm / KKm ** 2
        uci = u0_imag_h - divi * KXm / KKm ** 2
        vcr = u1_real_h - divr * KYm / KKm ** 2
        vci = u1_imag_h - divi * KYm / KKm ** 2

        ucr[0][0] = 0
        uci[0][0] = 0
        vcr[0][0] = 0
        vci[0][0] = 0

        e_kin = np.sum((ucr ** 2 + uci ** 2 + vcr ** 2 + vci ** 2)) * .5
        factor = np.sqrt(self.ic_energy / e_kin)
        ucr *= factor
        uci *= factor
        vcr *= factor
        vci *= factor

        norm = self.resolution / dx

        # backtransformation of u
        uc = ucr + uci * 1.0j
        ucf = np.fft.ifftn(uc, axes=(0, 1)) * norm
        u = ucf.real[None,...]

        # backtransformation of v
        vc = vcr + vci * 1.0j
        vcf = np.fft.ifftn(vc, axes=(0, 1)) * norm
        u = np.append(u, vcf.real[None, ...], axis=0)

        p = (u[0]*0)[None,...]
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
    def __init__(self, resolution, reynolds_number, mach_number, lattice, k0=20, ic_energy=0.5):
        self.k0 = k0
        self.ic_energy = ic_energy
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2*np.pi,
            characteristic_velocity_pu=1
        )



    def analytic_solution(self, x, t=0):
        return

    def initial_solution(self, x):
        dx = self.units.characteristic_length_pu / self.resolution

        ### Generate wavenumber vector
        kx = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        ky = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        kz = np.fft.fftfreq(self.resolution, d=1 / self.resolution)
        KX, KY, KZ = np.meshgrid(kx, ky, kz)
        KK = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        KK[0][0][0] = 1e-16

        ### Generate spectrum
        ek = (KK) ** 4 * np.exp(-2 * (KK / self.k0) ** 2)
        ek[0][0][0] = 0
        ek /= np.sum(ek)
        ek *= self.ic_energy

        # Forward transform random fields
        u = np.random.randn(self.resolution,self.resolution,self.resolution) + 0j
        v = np.random.randn(self.resolution,self.resolution,self.resolution) + 0j
        w = np.random.randn(self.resolution,self.resolution,self.resolution) + 0j

        uc = np.fft.fftn(u, axes=(0, 1, 2))
        vc = np.fft.fftn(v, axes=(0, 1, 2))
        wc = np.fft.fftn(w, axes=(0, 1, 2))

        # real parts
        ucr = uc.real
        vcr = vc.real
        wcr = wc.real
        # imaginary parts
        uci = uc.imag
        vci = vc.imag
        wci = wc.imag

        # no mean value
        ucr[0][0][0] = 0
        uci[0][0][0] = 0
        vcr[0][0][0] = 0
        vci[0][0][0] = 0
        wcr[0][0][0] = 0
        wci[0][0][0] = 0

        # scale with target energy at KK and divide by local energy at KK to force the spectrum
        u0_real_h = np.sqrt(2 / self.units.lattice.D * ek / (uci ** 2 + ucr ** 2 + 1.e-15)) * ucr
        u0_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (uci ** 2 + ucr ** 2 + 1.e-15)) * uci
        u1_real_h = np.sqrt(2 / self.units.lattice.D * ek / (vci ** 2 + vcr ** 2 + 1.e-15)) * vcr
        u1_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (vci ** 2 + vcr ** 2 + 1.e-15)) * vci
        u2_real_h = np.sqrt(2 / self.units.lattice.D * ek / (wci ** 2 + wcr ** 2 + 1.e-15)) * wcr
        u2_imag_h = np.sqrt(2 / self.units.lattice.D * ek / (wci ** 2 + wcr ** 2 + 1.e-15)) * wci
        u0_real_h[0][0][0] = 0
        u1_real_h[0][0][0] = 0
        u2_real_h[0][0][0] = 0
        u0_imag_h[0][0][0] = 0
        u1_imag_h[0][0][0] = 0
        u2_imag_h[0][0][0] = 0

        ### Remove divergence
        # modified wave number sin(k*dx) is used, as the gradient below uses second order cental differences
        # Modify if other schemes are used or use KX, KY if you don't know the modified wavenumber !!!
        KXm = np.sin(KX * dx) / dx
        KYm = np.sin(KY * dx) / dx
        KZm = np.sin(KZ * dx) / dx
        KKm = np.sqrt(KXm ** 2 + KYm ** 2 + KZm ** 2 + 1e-16)

        divr = (KXm * u0_real_h + KYm * u1_real_h + KZm * u2_real_h)
        divi = (KXm * u0_imag_h + KYm * u1_imag_h + KZm * u2_imag_h)

        ucr = u0_real_h - divr * KXm / KKm ** 2
        uci = u0_imag_h - divi * KXm / KKm ** 2
        vcr = u1_real_h - divr * KYm / KKm ** 2
        vci = u1_imag_h - divi * KYm / KKm ** 2
        wcr = u2_real_h - divr * KZm / KKm ** 2
        wci = u2_imag_h - divi * KZm / KKm ** 2

        ucr[0][0][0] = 0
        uci[0][0][0] = 0
        vcr[0][0][0] = 0
        vci[0][0][0] = 0
        wcr[0][0][0] = 0
        wci[0][0][0] = 0

        e_kin = np.sum((ucr ** 2 + uci ** 2 + vcr ** 2 + vci ** 2 + wcr ** 2 + wci ** 2)) * .5
        factor = np.sqrt(self.ic_energy / e_kin)
        ucr *= factor
        uci *= factor
        vcr *= factor
        vci *= factor
        wcr *= factor
        wci *= factor

        norm = self.resolution * dx ** (-2) * np.sqrt(self.units.characteristic_length_pu)

        # backtransformation of u
        uc = ucr + uci * 1.0j
        ucf = np.fft.ifftn(uc, axes=(0, 1, 2)) * norm
        u = ucf.real[None,...]

        # backtransformation of v
        vc = vcr + vci * 1.0j
        vcf = np.fft.ifftn(vc, axes=(0, 1, 2)) * norm
        u = np.append(u, vcf.real[None, ...], axis=0)

        # backtransformation of w
        wc = wcr + wci * 1.0j
        wcf = np.fft.ifftn(wc, axes=(0, 1, 2)) * norm
        u = np.append(u, wcf.real[None, ...], axis=0)

        #print(np.sum(u[0]**2+u[1]**2+u[2]**2)*.5*dx**3)
        p = (u[0]*0)[None,...]
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