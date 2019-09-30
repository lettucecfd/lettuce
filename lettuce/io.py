"""
Input/output routines.

TODO: Logging
TODO: VTK field/o
"""

import sys
import logging
import numpy as np
import torch
#from pyevtk.hl import *
import pyevtk.hl as vtk
from matplotlib import pyplot as plt

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

def write_png(filename, array2d):
    pass


def write_vtk(filename, resolution, field_data, id, uu, pp):
    # https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
    # https://anaconda.org/e3sm/evtk
    # https://pypi.org/project/pyevtk/
    a=1
    #data_1 = np.zeros((resolution[0], resolution[1], resolution[2]))
    #data_2 = np.zeros((resolution[0], resolution[1], resolution[2]))
    data_1 = np.zeros((1024, 1024, 1))
    data_2 = np.zeros((1024, 1024, 1))

    u = field_data.astype('float64')
    u = np.transpose(u, (2, 1, 0))

    q = pp.astype('float64')
    q = np.transpose(q, (2, 1, 0))

    print(id)

    vtk.gridToVTK("/Users/mariobedrunka/Downloads/data/" + "output_grid_" + id, np.arange(0, 1024), np.arange(0, 1024),
              np.arange(0, 1024), pointData={"ux": data_1, "uy": data_2, "rho": q})

    #raise NotImplementedError
class VTKReporter:
    """General VTK Reporter for velocity and pressure"""
    def __init__(self, lattice, flow, interval=50):
        assert hasattr(flow, "analytic_solution")
        self.lattice = lattice
        self.flow = flow
        self.interval = interval


    def __call__(self, i, t, f):
        if t % self.interval == 0:
            t = self.flow.units.convert_time_to_pu(t)
            #pref, uref = self.flow.analytic_solution(self.flow.grid, t=t)
            #pref = self.lattice.convert_to_tensor(pref)
            #uref = self.lattice.convert_to_tensor(uref)
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))

            resolution = torch.pow(torch.prod(self.lattice.convert_to_tensor(p.size())), 1 / self.lattice.D)

            write_vtk("output_vtk", resolution, self.lattice.convert_to_numpy(self.lattice.u(f)), str(t), self.lattice.convert_to_numpy(u), self.lattice.convert_to_numpy(p))

            #write_vtk("output_vtk", [self.flow.resolution * 2, self.flow.resolution, 1],
            #             self.lattice.convert_to_numpy(self.collision.lattice.u(self.f)),
            #             self.lattice.convert_to_numpy(self.collision.lattice.rho(self.f)), str(i))


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

