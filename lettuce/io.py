"""
Input/output routines.

TODO: Logging
"""

import sys
import numpy as np
import torch
import os
import pyevtk.hl as vtk
from lettuce.util import torch_gradient


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
        if t % self.interval == 0:
            t = self.flow.units.convert_time_to_pu(t)
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


class GenericStepReporter:
    """Abstract base class for reporters that print something every few iterations."""
    _parameter_name = None

    def __init__(self, lattice, flow, interval=1, starting_iteration=0, out=sys.stdout):
        self.lattice = lattice
        self.flow = flow
        self.starting_iteration = starting_iteration
        self.interval = interval
        self.out = [] if out is None else out
        print('steps    ', self._parameter_name)

    def __call__(self, i, t, f):
        if t % self.interval == 0:
            if t > self.starting_iteration:
                entry = [t, self.parameter_function(i,t,f)]
                if isinstance(self.out, list):
                    self.out.append(entry)
                else:
                    print(*entry, file=self.out)

    def parameter_function(self,i,t,f):
        return NotImplemented


class MaxUReporter(GenericStepReporter):
    """Reports the maximum velocity magnitude in the domain"""
    _parameter_name = 'Maximum velocity'

    def parameter_function(self,i,t,f):
        u0 = self.lattice.u(f)[0]
        u = u0 * u0
        if self.lattice.D > 1:
            u1 = self.lattice.u(f)[1]
            u += u1 * u1
            if self.lattice.D > 2:
                u2 = self.lattice.u(f)[2]
                u += u2 * u2
        return self.flow.units.convert_velocity_to_pu(torch.max(torch.sqrt(u)).cpu().numpy().item())


class EnergyReporter(GenericStepReporter):
    """Reports the kinetic energy """
    _parameter_name = 'Kinetic energy'

    def parameter_function(self,i,t,f):
        dx = self.flow.units.convert_length_to_pu(1.0)

        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)))
        kinE *= dx ** self.lattice.D
        return kinE.item()


class EnstrophyReporter(GenericStepReporter):
    """Reports the integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """
    _parameter_name = 'Enstrophy'

    def parameter_function(self,i,t,f):
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6).cpu().numpy()
        grad_u1 = torch_gradient(u1, dx=dx, order=6).cpu().numpy()
        vorticity = np.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6).cpu().numpy()
            vorticity += np.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity.item() * dx**self.lattice.D
