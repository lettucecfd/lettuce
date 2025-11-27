#TODO: WORK IN PROGRESS... -> this was copied from lettuce issue #248 and branch MaxBille:174-reporter-to-interrupt-simulation

import torch
import numpy as np

from lettuce import Reporter, Simulation
from ebb_simulation import EbbSimulation

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
        self.name = 'NaN'  # what is the name for?
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
                # the 1e10 is "a lot", but in a very unlikely case of a very long simulation,
                #  where num_steps >=1e10, this would not work...

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