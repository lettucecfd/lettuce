import os
from typing import List

import torch
import numpy as np

from ... import Reporter, Simulation
from .vtk_reporter import VTKReporter
from timeit import default_timer as timer

__all__ = ["NaNReporter", "HighMaReporter", "BreakableSimulation"]


class BreakableSimulation(Simulation):
    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter']):
        flow.context.use_native = False
        super().__init__(flow, collision, reporter)

    def __call__(self, num_steps: int):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        while self.flow.i < num_steps:
            self._collide_and_stream(self)
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)


class NaNReporter(Reporter):
    """reports any NaN and aborts the simulation"""
    # WARNING: too many NaNs in very large simulations can confuse torch and
    # trigger an error, when trying to create and store the nan_location
    # tensor.
    # ...to avoid this, leave outdir=None to omit creation and file-output of
    # nan_location. This will not impact the abortion of sim. by NaN_Reporter

    def __init__(self, interval=100, outdir=None, vtk=False):
        self.outdir = outdir
        self.vtk = vtk
        self.name = 'NaN'
        self.failed_iteration = None
        super().__init__(interval)

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i % self.interval == 0:
            if self.is_failed(simulation):
                self.failed_iteration = simulation.flow.i
                if self.outdir is not None:
                    self.outputs(simulation)

                print(
                    f'(!) ABORT MESSAGE: FailReporter detected {self.name}'
                    f' in f (interval = {self.interval}) at iteration '
                    f'{simulation.flow.i}. See log in {self.outdir} for '
                    f'details!')
                # telling simulation to abort simulation by setting i too high
                simulation.flow.i = int(simulation.flow.i + 1e10)

    def outputs(self, simulation: 'Simulation'):
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        my_file = open(f"{self.outdir}/{self.name}_reporter.txt",
                       "w")
        self.show_max(my_file, simulation)
        my_file.write(f"(!) {self.name} detected at \n")

        for location in self.locations_string(simulation):
            my_file.write(f"{location:6}   ")
        my_file.write("\n")
        for fail in self.failed_locations_list(simulation):
            for fail_dim in fail:
                my_file.write(f"{fail_dim:6}  ")
            my_file.write("\n")
        if len(self.failed_locations_list(simulation)) >= 100:
            my_file.write(f"(!) {self.name} detected for more "
                          f"than 100 values. Showing only first "
                          f"100 values\n")

        my_file.close()

        # write vtk output with u and p fields
        if self.vtk:
            vtkreporter = VTKReporter(
                1, filename_base=self.outdir + f"/{self.name}_fail")
            vtkreporter(simulation)

    def is_failed(self, simulation: 'Simulation') -> bool:
        # checks if any item of self.fails(simulation) is true
        return self.fails(simulation).any().item()

    def fails(self, simulation: 'Simulation') -> torch.Tensor:
        # returns a tensor with ([q],x,[y,z]) dimensions indicating whether
        # fail condition applies
        return torch.isnan(simulation.flow.f)

    def failed_locations_list(self, simulation: 'Simulation') -> np.ndarray:
        # getting fail locations (and possibly values)
        failed_locations_list = []
        for fail_dim in torch.where(self.fails(simulation)):
            failed_locations_list.append(
                simulation.context.convert_to_ndarray(fail_dim))
        if len(failed_locations_list[0]) > 100:
            return np.stack(failed_locations_list, axis=-1)[:100]
        return np.stack(failed_locations_list, axis=-1)

    def locations_string(self, simulation: 'Simulation') -> List[str]:
        locations = ["q", "x"]
        if simulation.flow.stencil.d > 1:
            locations.append("y")
        if simulation.flow.stencil.d > 2:
            locations.append("z")
        return locations

    def show_max(self, my_file, simulation: 'Simulation'):
        pass


def unravel_index(indices: torch.Tensor, shape: tuple[int, ...],
                  ) -> torch.Tensor:
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


class HighMaReporter(NaNReporter):
    """reports any Ma>0.3 and aborts the simulation"""
    def __init__(self, interval=100, outdir=None, vtk=False):
        super().__init__(interval, outdir, vtk)
        self.name = 'HighMa'

    def show_max(self, my_file, simulation: 'Simulation'):
        u = simulation.flow.u()
        ma = torch.norm(u, dim=0)/simulation.flow.stencil.cs
        index_max = torch.argmax(ma)
        index_max = unravel_index(index_max, ma.shape)
        ma = simulation.context.convert_to_ndarray(ma)
        max_ma = ma[
            index_max[0],
            index_max[1],
            index_max[2] if simulation.flow.stencil.d == 3 else None]
        my_file.write(
            f"Max. Ma{str(index_max.tolist())} = {max_ma}\n\n")

    def locations_string(self, simulation: 'Simulation') -> List[str]:
        locations = ["x"]
        if simulation.flow.stencil.d > 1:
            locations.append("y")
        if simulation.flow.stencil.d > 2:
            locations.append("z")
        locations.append("Ma")
        return locations

    def fails(self, simulation: 'Simulation') -> torch.Tensor:
        u = simulation.flow.u()
        ma = torch.norm(u, dim=0)/simulation.flow.stencil.cs
        return ma > 0.3

    def failed_locations_list(self, simulation: 'Simulation') -> np.ndarray:
        # getting fail locations
        failed_locations_list = []
        for fail_dim in torch.where(self.fails(simulation)):
            failed_locations_list.append(
                simulation.context.convert_to_ndarray(fail_dim))
        u = simulation.flow.u()
        ma = (torch.norm(u, dim=0)
              / simulation.flow.stencil.cs)[self.fails(simulation)]
        ma = simulation.context.convert_to_ndarray(ma)
        failed_locations_list.append(ma)
        if len(failed_locations_list[0]) >= 100:
            return np.stack(failed_locations_list, axis=-1)[:100]
        return np.stack(failed_locations_list, axis=-1)

    def first_100(self, simulation: 'Simulation'):
        u = simulation.flow.u()
        ma = torch.norm(u, dim=0)/simulation.flow.stencil.cs
        ma = simulation.context.convert_to_ndarray(ma)
        flat_fails = ma.ravel()
        k = 100
        indices = np.argpartition(-flat_fails, k)[:k]
        top_values = flat_fails[indices]
        sorted_indices = indices[np.argsort(-top_values)]
        failed_locations_list = np.array(
            np.unravel_index(sorted_indices,
                             self.fails(simulation).shape))
        return failed_locations_list[:100]
