import sys
from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np

from ... import Reporter, Flow
from .vtk_reporter import write_vtk

__all__ = ["NaNReporter", "HighMaReporter"]

class NaNReporter(Reporter):
    """reports any NaN and aborts the simulation"""
    # WARNING: too many NaNs in very large simulations can confuse torch and trigger an error, when trying to create and store the nan_location tensor.
    # ...to avoid this, leave outdir=None to omit creation and file-output of nan_location. This will not impact the abortion of sim. by NaN_Reporter

    def __init__(self, flow, lattice, n_target=None, t_target=None, interval=100, simulation=None, outdir=None, vtk=False, vtk_dir=None):
        self.flow = flow
        self.old = False
        if simulation is None:
            self.old = True
            self.n_target = n_target
        else:
            self.simulation = simulation
            self.n_target = simulation.n_steps_target
        self.lattice = lattice
        self.interval = interval
        self.t_target = t_target
        self.outdir = outdir
        self.vtk = vtk
        if vtk_dir is None:
            self.vtk_dir = self.outdir
        else:
            self.vtk_dir = vtk_dir
        #TMP vtk_dir = os.path.dirname(vtk_dir)
        #TMP if not os.path.isdir(directory):
        #TMP     os.mkdir(directory)

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            if torch.isnan(f).any():
                if self.lattice.D == 2 and self.outdir is not None:
                    q, x, y = torch.where(torch.isnan(f))
                    q = self.lattice.convert_to_numpy(q)
                    x = self.lattice.convert_to_numpy(x)
                    y = self.lattice.convert_to_numpy(y)
                    nan_location = np.stack((q, x, y), axis=-1)
                if self.lattice.D == 3 and self.outdir is not None:
                    q, x, y, z = torch.where(torch.isnan(f))
                    q = self.lattice.convert_to_numpy(q)
                    x = self.lattice.convert_to_numpy(x)
                    y = self.lattice.convert_to_numpy(y)
                    z = self.lattice.convert_to_numpy(z)
                    nan_location = np.stack((q, x, y, z), axis=-1)
                    if self.outdir is not None:
                        my_file = open(self.outdir, "w")
                        my_file.write(f"(!) NaN detected at (q,x,y,z):\n")
                        for _ in nan_location:
                            my_file.write(f"{_}\n")
                        my_file.close()
                        #print("(!) NaN detected at (q,x,y,z):", nan_location)

                if self.old:
                    # backwards compatibility for simulation class w/o abort-message-functionality
                    print("(!) NaN detected in time step", i, "of", self.n_target, "(interval:", self.interval, ")")
                    sys.exit()
                else:
                    self.simulation.abort_condition = 2  # telling simulation to abort simulation
                    self.simulation.abort_message = f'(!) ABORT MESSAGE: NaNReporter detected NaN in f (NaNReporter.interval = {self.interval}). See NaNReporter log for details!'
                    # print("(!) NaN detected in time step", i, "of", self.simulation.n_steps_target, "(interval:", self.interval, ")")
                    # print("(!) Aborting simulation at t_PU", self.flow.units.convert_time_to_pu(i), "of", self.flow.units.convert_time_to_pu(self.simulation.n_steps_target))

                # write vtk output with u and p fields to vtk_dir, if vtk_dir is not None
                if self.vtk_dir is not None and self.vtk:
                    point_dict = dict()
                    u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
                    p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))
                    if self.lattice.D == 2:
                        point_dict["p"] = self.lattice.convert_to_numpy(p[0, ..., None])
                        for d in range(self.lattice.D):
                            point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ..., None])
                    else:
                        point_dict["p"] = self.lattice.convert_to_numpy(p[0, ...])
                        for d in range(self.lattice.D):
                            point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])
                    write_vtk(point_dict, i, self.vtk_dir+"/nan_frame")


def unravel_index(indices: torch.Tensor, shape: tuple[int, ...], ) -> torch.Tensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


class HighMaReporter(Reporter):
    """reports any Ma>0.3 and aborts the simulation"""

    def __init__(self, flow, lattice, n_target=None, t_target=None, interval=100, simulation=None, outdir=None, vtk=False, vtk_dir=None):
        self.flow = flow
        self.old = False
        if simulation is None:
            self.old = True
            self.n_target = n_target
        else:
            self.simulation = simulation
            self.n_target = simulation.n_steps_target
        self.lattice = lattice
        self.interval = interval
        self.t_target = t_target
        self.outdir = outdir
        self.vtk = vtk
        if vtk_dir is None:
            self.vtk_dir = self.outdir
        else:
            self.vtk_dir = vtk_dir

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            u = self.lattice.u(f)
            ma = torch.norm(u, dim=0)/self.lattice.cs
            # return torch.tensor([u_mag.max(), indices], device=u.device)

            high_ma_locations = torch.where(ma > 0.3, True, False)

            if high_ma_locations.any():
                if self.lattice.D == 2 and self.outdir is not None:
                    x, y = torch.where(high_ma_locations)
                    more_than_100 = False
                    if x.shape[0] < 100:
                        x = self.lattice.convert_to_numpy(x)
                        y = self.lattice.convert_to_numpy(y)
                        high_ma_locations = np.stack((x, y), axis=-1)
                    else:
                        more_than_100 = True
                if self.lattice.D == 3 and self.outdir is not None:
                    x, y, z = torch.where(high_ma_locations)
                    more_than_100 = False
                    if x.shape[0] < 100:
                        x = self.lattice.convert_to_numpy(x)
                        y = self.lattice.convert_to_numpy(y)
                        z = self.lattice.convert_to_numpy(z)
                        high_ma_locations = np.stack((x, y, z), axis=-1)
                    else:
                        more_than_100 = True
                if self.outdir is not None:
                    my_file = open(self.outdir+"/HighMa_reporter.txt", "w")

                    my_file.write(f"(!) Ma > 0.3 detected , Maximum at (x,y,[z]):\n")
                    index_max = torch.argmax(ma)
                    index_max = unravel_index(index_max, ma.shape)
                    ma = self.lattice.convert_to_numpy(ma)
                    index_max = self.lattice.convert_to_numpy(index_max)
                    my_file.write(f" Ma {str(list(index_max))} = {ma[index_max[0], index_max[1], index_max[2] if self.lattice.D == 3 else None]}\n\n")
                    #TODO: write PU coordinates as well. a) in seperate file, b) same file below, c) same file new column "table style"
                    if not more_than_100:
                        my_file.write(f"(!) Ma > 0.3 detected at (x,y,[z]):\n")
                        for _ in high_ma_locations:
                            my_file.write(f"Ma {_} = {ma[_[0], _[1], _[2] if self.lattice.D == 3 else None]}\n")
                    else:
                        flat_ma = ma.ravel()
                        k=100
                        indices = np.argpartition(-flat_ma, k)[:k]
                        top_values = flat_ma[indices]
                        sorted_indices = indices[np.argsort(-top_values)]
                        sorted_values = flat_ma[sorted_indices]
                        original_indices = np.array(np.unravel_index(sorted_indices, ma.shape))
                        print(original_indices)
                        print(original_indices.shape[0], original_indices.shape[1])
                        my_file.write(f"(!) Ma > 0.3 detected for more than 100 values. Showing top 100 values:\n")
                        for _ in range(original_indices.shape[1]):
                            my_file.write(f"Ma {original_indices[:,_]} = {ma[original_indices[0,_], original_indices[1,_], original_indices[2,_] if self.lattice.D == 3 else None]:15.4f}\n")
                    my_file.close()

                if self.old:
                    print("(!) Ma > 0.3 detected in time step", i, "of", self.n_target, "(interval:", self.interval, ")")
                    sys.exit()
                else:
                    self.simulation.abort_condition = 3  # telling simulation to abort simulation
                    self.simulation.abort_message = f'(!) ABORT MESSAGE: Ma > 0.3 detected (HighMaReporter.interval = {self.interval}). See HighMaReporter log for details!'
                    #print("(!) NaN detected in time step", i, "of", self.simulation.n_steps_target, "(interval:", self.interval, ")")
                    #print("(!) Aborting simulation at t_PU", self.flow.units.convert_time_to_pu(i), "of", self.flow.units.convert_time_to_pu(self.simulation.n_steps_target))

                # write vtk output with u and p fields to vtk_dir, if vtk_dir is not None
                if self.vtk_dir is not None and self.vtk:
                    point_dict = dict()
                    u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
                    p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))
                    if self.lattice.D == 2:
                        point_dict["p"] = self.lattice.convert_to_numpy(p[0, ..., None])
                        for d in range(self.lattice.D):
                            point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ..., None])
                    else:
                        point_dict["p"] = self.lattice.convert_to_numpy(p[0, ...])
                        for d in range(self.lattice.D):
                            point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])
                    write_vtk(point_dict, i, self.vtk_dir + "/highMa_frame")