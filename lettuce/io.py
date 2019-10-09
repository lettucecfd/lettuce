"""
Input/output routines.

TODO: Logging
TODO: VTK field/o
"""

import sys
import logging
import numpy as np
import torch


def write_image(filename, array2d):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.imshow(array2d)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(filename)


def write_vtk(filename, grid, field_data):
    # https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
    # https://anaconda.org/e3sm/evtk
    # https://pypi.org/project/pyevtk/
    raise NotImplementedError


class ErrorReporter:
    """Reports numerical errors with respect to analytic solution."""
    def __init__(self, lattice, flow, interval=1, out=sys.stdout):
        assert hasattr(flow, "analytic_solution")
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.out = [] if out is None else out
        if not isinstance(self.out, list):
            print("#error_u         error_p", file=self.out)

    def __call__(self, i, t, f):
        if t % self.interval == 0:
            t = self.flow.units.convert_time_to_pu(t)
            pref, uref = self.flow.analytic_solution(self.flow.grid, t=t)
            pref = self.lattice.convert_to_tensor(pref)
            uref = self.lattice.convert_to_tensor(uref)
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))

            resolution = torch.pow(torch.prod(self.lattice.convert_to_tensor(p.size())),1/self.lattice.D)

            err_u = torch.norm(u-uref)/resolution
            err_p = torch.norm(p-pref)/resolution

            if isinstance(self.out, list):
                self.out.append([err_u.item(), err_p.item()])
            else:
                print(err_u.item(), err_p.item(), file=self.out)


class EnergyReporter:
    """Reports the kinetic energy with respect to analytic solution."""
    def __init__(self, lattice, flow, interval=1, out=sys.stdout):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.out = [] if out is None else out
        if not isinstance(self.out, list):
            print("time      kinetic energy", file=self.out)

    def __call__(self, i, t, f):
        if t % self.interval == 0:
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            dx = self.flow.units.convert_length_to_pu(1.0)

            kinE=torch.sum(self.lattice.energy(f))
            kinE*=dx**self.lattice.D
            if isinstance(self.out, list):
                self.out.append([t, kinE.item()])
            else:
                print(t, kinE.item(), file=self.out)



