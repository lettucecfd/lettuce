import torch

import os
from typing import List

from ... import Reporter, Simulation, BreakableSimulation
from .vtk_reporter import write_vtk

__all__ = ["FailureReporterBase", "NaNReporter", "HighMaReporter"]

# Angenommene Superklasse (falls sie nicht importiert werden kann)
# class Reporter: ...


class FailureReporterBase(Reporter):
    """Abstrakte Basis für Reporter, die bei Fehlern (NaN, HighMa) loggen."""

    def __init__(self, interval, k=100, outdir=None, vtk_out=False):
        super().__init__(interval)
        self.k = k
        self.outdir = outdir
        self.name = "FailureReporter"
        self.vtk_out = vtk_out
        self.failed_iteration = None

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i % self.interval == 0:
            if self.is_failed(simulation):
                results = self.get_results(simulation)
                self.failed_iteration = simulation.flow.i

                if self.outdir is not None:
                    self._write_log(simulation, results)
                    self.save_vtk(simulation)

                print(
                    f'(!) ABORT MESSAGE: {self.name}Reporter detected {self.name}'
                    f' (reporter-interval = {self.interval}) at iteration '
                    f'{simulation.flow.i}. See log in {self.outdir} for '
                    f'details!')
                # telling simulation to abort simulation by setting i too high
                simulation.flow.i = int(simulation.flow.i + 1e10)
                # TODO: maybe make this more robust with a failed-flag in simulation and not rely on flow.i to be high/low?
                # the 1e10 is "a lot", but in a very unlikely case of a very long simulation,
                #  where num_steps >=1e10, this would not work...

    def _get_top_failures(self, mask, values):
        """Extrahiert Koordinaten und Werte."""
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

    def _write_log(self, simulation, results):
        """Schreibt die Ergebnisse in die Datei."""
        # Annahme: simulation hat Zugriff auf ein Dateiobjekt oder Pfad
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        # Pfad zusammenbauen
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

    def locations_string(self, simulation: 'Simulation') -> List[str]:
        locations = ["q", "x"]
        if simulation.flow.stencil.d > 1:
            locations.append("y")
        if simulation.flow.stencil.d > 2:
            locations.append("z")
        return locations

    def save_vtk(self, simulation):
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
                point_dict[f"u{'xyz'[d]}"] = simulation.flow.context.convert_to_ndarray(
                    u[d, ...])
        write_vtk(point_dict, simulation.flow.i, self.outdir + f"/{self.name}_frame")

    def is_failed(self, simulation):
        return torch.isnan(simulation.flow.f).any()

    def get_results(self, simulation):
        nan_mask = torch.isnan(simulation.flow.f)
        return self._get_top_failures(nan_mask, simulation.flow.f)


class NaNReporter(FailureReporterBase):

    def __init__(self, interval, k=100, outdir=None, vtk_out=False):

        super().__init__(interval, k, outdir, vtk_out)
        self.name = "NaN"

    def is_failed(self, simulation):
        return torch.isnan(simulation.flow.f).any()

    def get_results(self, simulation):
        nan_mask = torch.isnan(simulation.flow.f)
        return self._get_top_failures(nan_mask, simulation.flow.f)

    def locations_string(self, simulation: 'Simulation') -> List[str]:
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


    def is_failed(self, simulation):
        u = simulation.flow.u()
        # Machzahl auf GPU berechnen
        ma = torch.norm(u, dim=0) / simulation.flow.stencil.cs
        return (ma > self.threshold).any()

    def get_results(self, simulation):
        u = simulation.flow.u()
        # Machzahl auf GPU berechnen
        ma = torch.norm(u, dim=0) / simulation.flow.stencil.cs
        mask = ma > self.threshold
        return self._get_top_failures(mask, ma)

    def locations_string(self, simulation: 'Simulation') -> List[str]:
        locations = ["x"]
        if simulation.flow.stencil.d > 1:
            locations.append("y")
        if simulation.flow.stencil.d > 2:
            locations.append("z")
        locations.append("Ma")
        return locations