import torch

import os
from typing import List, Tuple
from abc import ABC, abstractmethod

from ... import Reporter, BreakableSimulation
from .vtk_reporter import write_vtk

__all__ = ["FailureReporterBase", "NaNReporter", "HighMaReporter"]

class FailureReporterBase(Reporter, ABC):
    """
    abstract base class for reporters that detect a failing simulation, due
    to conditions like NaN, high Mach number etc.
    - relies on BreakableSimulation class (!)
    """
    #TODO (optional): make STOPPING the simulation optional
    # -> for example the HighMa-Reporter only reports high Mach numbers and
    # their location, but the simulation just continues normally.
    # This could be useful and "ok" in some cases: EXAMPLE, in the event
    # of a settling period (transient high velocities) at the beginning
    # of the run.

    def __init__(self, interval, k=100, outdir=None, vtk_out=False):
        super().__init__(interval)
        self.k = k
        self.outdir = outdir
        self.name = "FailureReporter"
        self.vtk_out = vtk_out
        self.failed_iteration = None

    def __call__(self, simulation: 'BreakableSimulation'):
        if simulation.flow.i % self.interval == 0:
            if self.is_failed(simulation):
                results = self.get_results(simulation)
                self.failed_iteration = simulation.flow.i

                if self.outdir is not None:
                    self._write_log(simulation, results)
                    self.save_vtk(simulation)

                print(
                    f'(!) ABORT MESSAGE: {self.name}Reporter detected '
                    f'{self.name} (reporter-interval = {self.interval}) '
                    f'at iteration {simulation.flow.i}. '
                    f'See log in {self.outdir} for details!')
                # telling simulation to abort simulation by setting i too high
                simulation.flow.i = int(simulation.flow.i + 1e10)
                # TODO: make this more robust with a failed-flag in simulation
                #  and not rely on flow.i to be high "enough" to be higher than
                #  the target steps number given to simulation(steps)?
                # the 1e10 is "a lot", but in a very unlikely case of a very long simulation,
                #  where num_steps >=1e10, this would not work...

    def _get_top_failures(self, mask, values) -> List[Tuple]:
        """extract coordinates and values at nodes (mask);
            returns list of (pos, val) tuples"""
        failed_values = values[mask]
        all_coords = torch.nonzero(mask)
        num_to_extract = min(self.k, failed_values.numel())

        if torch.isnan(failed_values).any():
            top_indices = torch.arange(num_to_extract, device=values.device)
        else:
            _, top_indices = torch.topk(failed_values, k=num_to_extract,
                                        largest=True, sorted=True)

        top_coords = all_coords[top_indices].cpu().numpy()
        top_values = failed_values[top_indices].cpu().numpy()

        return [
            (list(c.astype(int)), float(v))
            for c, v in zip(top_coords, top_values)
        ]

    def _write_log(self, simulation: 'BreakableSimulation', results):
        """writes results to file and logs flow.i"""

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        filepath = os.path.join(self.outdir, f"{self.name}_reporter.log")
        with open(filepath, "w") as file:
            file.write(f"(!) {self.name} detected in step {simulation.flow.i} "
                       f"at following locations (top {self.k} listed):\n")
            file.write("     ")
            for location in self.locations_string(simulation):
                file.write(f"{location:6}   ")
            file.write("\n")
            for pos, val in results:
                line=""
                for pos_i in pos:
                    line = line + f"{int(pos_i):6}   "
                file.write(f"{line:<20} | {val:<15.6f}\n")
            file.write("\n")

    def save_vtk(self, simulation: 'BreakableSimulation'):
        """saves vtk file to outdir"""
        point_dict = dict()
        u = simulation.flow.u_pu
        p = simulation.flow.p_pu
        if simulation.flow.stencil.d == 2:
            point_dict["p"] = simulation.flow.context.convert_to_ndarray(p[0, ..., None])
            for d in range(simulation.flow.stencil.d):
                point_dict[f"u{'xyz'[d]}"] = simulation.flow.context.convert_to_ndarray(
                    u[d, ..., None])
        else:
            point_dict["p"] = simulation.flow.context.convert_to_ndarray(p[0, ...])
            for d in range(simulation.flow.stencil.d):
                point_dict[f"u{'xyz'[d]}"] = simulation.flow.context.convert_to_ndarray(u[d, ...])
        write_vtk(point_dict, simulation.flow.i, self.outdir + f"/{self.name}_frame")

    @abstractmethod
    def locations_string(self, simulation: 'BreakableSimulation') -> List[str]:
        ...

    @abstractmethod
    def is_failed(self, simulation: 'BreakableSimulation'):
        """checks if simulation meets criterion"""
        ...

    @abstractmethod
    def get_results(self, simulation: 'BreakableSimulation'):
        """calls specific method to create list of locations at which
        simulation failed"""
        ...


class NaNReporter(FailureReporterBase):

    def __init__(self, interval, k=100, outdir=None, vtk_out=False):
        super().__init__(interval, k, outdir, vtk_out)
        self.name = "NaN"

    def is_failed(self, simulation: 'BreakableSimulation'):
        return torch.isnan(simulation.flow.f).any()

    def get_results(self, simulation: 'BreakableSimulation'):
        nan_mask = torch.isnan(simulation.flow.f)
        return self._get_top_failures(nan_mask, simulation.flow.f)

    def locations_string(self, simulation: 'BreakableSimulation') -> List[str]:
        """create locations string as a header for the table of locations and
        values in the output"""
        locations = ["q", "x"]
        if simulation.flow.stencil.d > 1:
            locations.append("y")
        if simulation.flow.stencil.d > 2:
            locations.append("z")
        return locations


class HighMaReporter(FailureReporterBase):
    def __init__(self, interval, threshold=0.3, k=100, outdir=None, vtk_out=False):
        super().__init__(interval, k, outdir, vtk_out)
        self.threshold = threshold
        self.name = "HighMa"


    def is_failed(self, simulation: 'BreakableSimulation'):
        u = simulation.flow.u()
        ma = torch.norm(u, dim=0) / simulation.flow.stencil.cs
        return (ma > self.threshold).any()

    def get_results(self, simulation: 'BreakableSimulation'):
        u = simulation.flow.u()
        ma = torch.norm(u, dim=0) / simulation.flow.stencil.cs
        mask = ma > self.threshold
        return self._get_top_failures(mask, ma)

    def locations_string(self, simulation: 'BreakableSimulation') -> List[str]:
        """create locations string as a header for the table of locations and
                values in the output"""
        locations = ["x"]
        if simulation.flow.stencil.d > 1:
            locations.append("y")
        if simulation.flow.stencil.d > 2:
            locations.append("z")
        locations.append("Ma")
        return locations