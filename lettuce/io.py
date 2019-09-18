"""
Input/output routines.

TODO: Logging
TODO: VTK field/o
"""

import sys
import logging
import numpy as np
import torch
from pyevtk.hl import *
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


def write_vtk(filename, resolution, field_data, id):
    # https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
    # https://anaconda.org/e3sm/evtk
    # https://pypi.org/project/pyevtk/
    data_1 = np.zeros((resolution[0], resolution[1], resolution[2]))
    data_2 = np.zeros((resolution[0], resolution[1], resolution[2]))

    u = field_data.astype('float64')
    u = np.transpose(u, (2, 1, 0))


    data_1[:, :, 0] = u[:, :, 0]
    data_2[:, :, 0] = u[:, :, 1]

    gridToVTK("/Users/mariobedrunka/Documents/10_science/10_lattice_boltzmann/10_simulation/10_lettuce/data/" + "output_grid_" + id, np.arange(0, resolution[0]), np.arange(0, resolution[1]),
              np.arange(0, resolution[2]), pointData={"ux": data_1, "uy": data_2})

    #raise NotImplementedError


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

