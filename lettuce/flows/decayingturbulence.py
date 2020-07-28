"""
Decaying turublence in 2D.
"""

import numpy as np

from lettuce.unit import UnitConversion


class DecayingTurbulence2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2*np.pi,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        nu = self.units.viscosity_pu
        u = np.array([np.cos(x[0]) * np.sin(x[1]) * np.exp(-2*nu*t), -np.sin(x[0]) * np.cos(x[1]) * np.exp(-2*nu*t)])
        p = np.array([0.25 * (np.cos(2*x[0]) + np.cos(2*x[1])) * np.exp(-4 * nu * t)])
        return p, u

    def initial_solution(self, x, k0=50):
        dx = x[0][0, 1] - x[0][0, 0]

        # Generate wavenumber function
        kx = np.fft.fftfreq(self.resolution, d=1. / self.resolution)
        ky = np.fft.fftfreq(self.resolution, d=1. / self.resolution)
        kk = np.sqrt(kx ** 2 + ky[:, np.newaxis] ** 2)
        kk[0][0] = 1e-6

        #  Generate random gaussian zero mean phase function
        xi = np.random.random_sample((self.resolution, self.resolution)) * 2 * np.pi
        eta = np.random.random_sample((self.resolution, self.resolution)) * 2 * np.pi
        phase = np.zeros((self.resolution, self.resolution), dtype='complex128')
        phase[1:self.resolution // 2, 1:self.resolution // 2] = - xi[1:self.resolution // 2, 1:self.resolution // 2] \
                                                      - eta[1:self.resolution // 2, 1:self.resolution // 2]
        phase[1:self.resolution // 2, self.resolution // 2:-1] = -xi[1:self.resolution // 2, self.resolution // 2:-1] \
                                                       + eta[1:self.resolution // 2, self.resolution // 2:-1]
        phase[self.resolution // 2:-1, 1:self.resolution // 2] = xi[self.resolution // 2:-1, 1:self.resolution // 2] \
                                                       - eta[self.resolution // 2:-1, 1:self.resolution // 2]
        phase[self.resolution // 2:-1, self.resolution // 2:-1] = xi[self.resolution // 2:-1, self.resolution // 2:-1] \
                                                        + eta[self.resolution // 2:-1, self.resolution // 2:-1]
        ## Energy spectrum in fourier space
        s0 = 3
        a0 = ((2 * s0 + 1) ** (s0 + 1)) / ((2 ** s0) * np.math.factorial(s0))
        ek = a0 / 2 / k0 * (kk / k0) ** (2 * s0 + 1) * np.exp(-(s0 + 0.5) * (kk / k0) ** 2)

        ## Vorticity function in fourier space
        w_sp = np.sqrt((kk * ek / np.pi)) * np.exp(1j * phase)

        ## (Inverse) fourier transformation
        w_pu = np.fft.ifftn(w_sp, axes=(0, 1)).real
        psi_sp = w_sp / (kk ** 2)
        psi_pu = np.fft.ifftn(psi_sp, axes=(0, 1)).real
        grad_psi_pu = np.gradient(psi_pu, dx)

        u_pu = np.array([grad_psi_pu[1]])
        p_pu = u_pu*0
        u_pu = np.append(u_pu, [-grad_psi_pu[0]], axis=0)

        return p_pu, u_pu

    @property
    def grid(self):

        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y)

    @property
    def boundaries(self):
        return []