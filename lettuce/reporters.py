"""
Input/output routines.
TODO: Logging
"""

import sys
import warnings
import os
import numpy as np
import torch
import pyevtk.hl as vtk

__all__ = [
    "write_image", "write_vtk", "VTKReporter", "ObservableReporter", "ErrorReporter"
]


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


def write_vtk(point_dict, id=0, filename_base="./data/output"):
    vtk.gridToVTK(f"{filename_base}_{id:08d}",
                  np.arange(0, point_dict["p"].shape[0]),
                  np.arange(0, point_dict["p"].shape[1]),
                  np.arange(0, point_dict["p"].shape[2]),
                  pointData=point_dict)


class VTKReporter:
    """General VTK Reporter for velocity and pressure"""

    def __init__(self, lattice, flow, interval=50, filename_base="./data/output"):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.filename_base = filename_base
        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self.point_dict = dict()

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))
            if self.lattice.D == 2:
                self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ..., None])
                for d in range(self.lattice.D):
                    self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ..., None])
            else:
                self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ...])
                for d in range(self.lattice.D):
                    self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])
            write_vtk(self.point_dict, i, self.filename_base)

    def output_mask(self, no_collision_mask):
        """Outputs the no_collision_mask of the simulation object as VTK-file with range [0,1]
        Usage: vtk_reporter.output_mask(simulation.no_collision_mask)"""
        point_dict = dict()
        if self.lattice.D == 2:
            point_dict["mask"] = self.lattice.convert_to_numpy(no_collision_mask)[..., None].astype(int)
        else:
            point_dict["mask"] = self.lattice.convert_to_numpy(no_collision_mask).astype(int)
        vtk.gridToVTK(self.filename_base + "_mask",
                      np.arange(0, point_dict["mask"].shape[0]),
                      np.arange(0, point_dict["mask"].shape[1]),
                      np.arange(0, point_dict["mask"].shape[2]),
                      pointData=point_dict)


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
        if i % self.interval == 0:
            pref, uref = self.flow.analytic_solution(self.flow.grid, t=t)
            pref = self.lattice.convert_to_tensor(pref)
            uref = self.lattice.convert_to_tensor(uref)
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))

            resolution = torch.pow(torch.prod(self.lattice.convert_to_tensor(p.size())), 1 / self.lattice.D)

            err_u = torch.norm(u - uref) / resolution ** (self.lattice.D / 2)
            err_p = torch.norm(p - pref) / resolution ** (self.lattice.D / 2)

            if isinstance(self.out, list):
                self.out.append([err_u.item(), err_p.item()])
            else:
                print(err_u.item(), err_p.item(), file=self.out)


class ObservableReporter:
    """A reporter that prints an observable every few iterations.

    Examples
    --------
    Create an Enstrophy reporter.

    >>> from lettuce import TaylorGreenVortex3D, Enstrophy, D3Q27, Lattice
    >>> lattice = Lattice(D3Q27, device="cpu")
    >>> flow = TaylorGreenVortex(50, 300, 0.1, lattice)
    >>> enstrophy = Enstrophy(lattice, flow)
    >>> reporter = ObservableReporter(enstrophy, interval=10)
    >>> # simulation = ...
    >>> # simulation.reporters.append(reporter)
    """

    def __init__(self, observable, interval=1, out=sys.stdout):
        self.observable = observable
        self.interval = interval
        self.out = [] if out is None else out
        self._parameter_name = observable.__class__.__name__
        print('steps    ', 'time    ', self._parameter_name)

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            observed = self.observable.lattice.convert_to_numpy(self.observable(f))
            assert len(observed.shape) < 2
            if len(observed.shape) == 0:
                observed = [observed.item()]
            else:
                observed = observed.tolist()
            entry = [i, t] + observed
            if isinstance(self.out, list):
                self.out.append(entry)
            else:
                print(*entry, file=self.out)