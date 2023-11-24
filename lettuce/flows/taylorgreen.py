"""
Taylor-Green vortex in 2D and 3D.
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.boundary import FlippedBoundary
from lettuce.boundary import TGV3D
from lettuce.boundary import superTGV3D
from lettuce.boundary import newsuperTGV3D
class TaylorGreenVortex2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        nu = self.units.viscosity_pu
        u = np.array([np.cos(x[0]) * np.sin(x[1]) * np.exp(-2 * nu * t),
                      -np.sin(x[0]) * np.cos(x[1]) * np.exp(-2 * nu * t)])
        p = -np.array([0.25 * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * np.exp(-4 * nu * t)])

        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)

    @property
    def grid(self):
        x,dx = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False,retstep=True)
        x=x+dx/2
        y,dy = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False,retstep=True)
        y=y+dy/2

        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []


class TaylorGreenVortex3D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution/(2*np.pi) , characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):
        u = np.array([
            np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2]),
            -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2]),
            np.zeros_like(np.sin(x[0]))
        ])
        p = np.array([1 / 16. * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * (np.cos(2 * x[2]) + 2)])
        return p, u

    @property
    def grid(self):
        x,dx = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False, retstep=True)
        x=x+dx/2
        y,dy = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False, retstep=True)
        y=y+dy/2
        z,dz = np.linspace(np.pi/2, 5/2 * np.pi, num=self.resolution, endpoint=False,retstep=True)
        z=z+dz/2
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        return []


class ReducedTaylorGreenVortex2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=np.pi,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        nu = self.units.viscosity_pu
        #u = np.array([np.cos(x[0]) * np.sin(x[1]) * np.exp(-2 * nu * t),
                      #-np.sin(x[0]) * np.cos(x[1]) * np.exp(-2 * nu * t)])
        #p = -np.array([0.25 * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * np.exp(-4 * nu * t)])
        u = np.array([np.cos(x[0]) * np.sin(x[1]) * np.exp(-2 * nu * t),
                      -np.sin(x[0]) * np.cos(x[1]) * np.exp(-2 * nu * t)])
        p = -np.array([0.25 * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * np.exp(-4 * nu * t)])
        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)

    @property
    def grid(self):
        x, dx = np.linspace(np.pi/2,5*np.pi/2, num=(self.resolution), endpoint=False, retstep=True)
        x = x + dx / 2
        #x = np.concatenate(([-dx / 2], x, [np.pi + dx / 2]))
        y, dy = np.linspace(np.pi/2,5*np.pi/2, num=(self.resolution), endpoint=False, retstep=True)
        y = y + dy / 2
        #y =np.concatenate(([-dy / 2], y, [np.pi + dy / 2]))

        #x = np.linspace(0, np.pi, num=(self.resolution), endpoint=True)
        #y = np.linspace(0, np.pi, num=(self.resolution), endpoint=True)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        #mask=np.zeros(((self.resolution),(self.resolution)),dtype=bool)
        #mask[1:-1, 1:-1] = False
        #mask[[0, -1], :] = True
        #mask[:, [0, -1]] = True
        boundary = FlippedBoundary()
        return [boundary]


class ReducedTaylorGreenVortex3D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution / (np.pi), characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):
        u = np.array([
            np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2]),
            -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2]),
            np.zeros_like(np.sin(x[0]))
        ])
        p = np.array([1 / 16. * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * (np.cos(2 * x[2]) + 2)])
        return p, u

    @property
    def grid(self):
        x,dx = np.linspace(0, np.pi, num=self.resolution, endpoint=False, retstep=True)
        x=x+dx/2
        y,dy = np.linspace(0, np.pi, num=self.resolution, endpoint=False, retstep=True)
        y=y+dy/2
        z,dz = np.linspace(np.pi/2, 3/2*np.pi, num=self.resolution, endpoint=False, retstep=True)
        z=z+dz/2
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        boundary=TGV3D(lattice=self.units.lattice)
        return [boundary]


class SuperReducedTaylorGreenVortex3D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution / (1/2*np.pi), characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):
        u = np.array([
            np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2]),
            -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2]),
            np.zeros_like(np.sin(x[0]))
        ])
        p = np.array([1 / 16. * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * (np.cos(2 * x[2]) + 2)])
        return p, u

    @property
    def grid(self):
        x,dx = np.linspace(0, np.pi/2, num=self.resolution, endpoint=False, retstep=True)
        x=x+dx/2
        y,dy = np.linspace(0, np.pi/2, num=self.resolution, endpoint=False, retstep=True)
        y=y+dy/2
        z,dz = np.linspace(np.pi/2, np.pi, num=self.resolution, endpoint=False, retstep=True)
        z=z+dz/2
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        boundary=newsuperTGV3D(lattice=self.units.lattice)
        return [boundary]