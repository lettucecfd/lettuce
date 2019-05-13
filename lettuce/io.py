"""
Input/output routines.

TODO: Logging
TODO: VTK i/o
"""

import sys
import logging
import numpy as np
import torch
from matplotlib import pyplot as plt


def write_png(filename, array2d):
    pass


def write_vtk(filename, grid, field_data):
    # https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
    # https://anaconda.org/e3sm/evtk
    # https://pypi.org/project/pyevtk/
    raise NotImplementedError


class ErrorReporter(object):
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

            num_gridpoints = torch.prod(self.lattice.convert_to_tensor(p.size()))

            err_u = np.linalg.norm(u-uref)/num_gridpoints
            err_p = np.linalg.norm(p-pref)/num_gridpoints

            if isinstance(self.out, list):
                self.out.append([err_u, err_p])
            else:
                print(err_u, err_p, file=self.out)

