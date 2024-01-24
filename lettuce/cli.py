# -*- coding: utf-8 -*-

"""Console script for lettuce.
To get help for terminal commands, open a console and type:

>>>  lettuce --help

"""

import matplotlib.pyplot as plt

import sys
import cProfile
import pstats

import click
import torch
import lettuce
import numpy as np
from timeit import default_timer as timer

from lettuce import BGKCollision, StandardStreaming, Lattice, D2Q9, Pipeline, StandardRead, Write, TRTCollision, Read, StandardWrite
from lettuce import __version__ as lettuce_version

from lettuce import TaylorGreenVortex2D, Simulation, ErrorReporter, VTKReporter
from lettuce.flows import flow_by_name
from lettuce.force import Guo


@click.group()
@click.version_option(version=lettuce_version)
@click.option("--cuda/--no-cuda", default=True, help="Use cuda (default=True).")
@click.option("-i", "--gpu-id", type=int, default=0, help="Device ID of the GPU (default=0).")
@click.option("-p", "--precision", type=click.Choice(["half", "single", "double"]), default="double",
              help="Numerical Precision; 16, 32, or 64 bit per float (default=double).")
@click.pass_context  # pass parameters to sub-commands
def main(ctx, cuda, gpu_id, precision):
    """Pytorch-accelerated Lattice Boltzmann Solver
    """
    ctx.obj = {'device': None, 'dtype': None}
    if cuda:
        if not torch.cuda.is_available():
            print("CUDA not found.")
            raise click.Abort
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = torch.device("cpu")
    dtype = {"half": torch.half, "single": torch.float, "double": torch.double}[precision]

    ctx.obj['device'] = device
    ctx.obj['dtype'] = dtype


@main.command()
@click.option("-s", "--steps", type=int, default=10, help="Number of time steps.")
@click.option("-r", "--resolution", type=int, default=1024, help="Grid Resolution")
@click.option("-o", "--profile-out", type=str, default="",
              help="File to write profiling information to (default=""; no profiling information gets written).")
@click.option("-f", "--flow", type=click.Choice(flow_by_name.keys()), default="taylor2D")
@click.option("-v", "--vtk-out", type=str, default="",
              help="VTK file basename to write the velocities and densities to (default=""; no info gets written).")
@click.option("--use-native/--use-no-native", default=True, help="whether to use the native implementation or not.")
@click.pass_context  # pass parameters to sub-commands
def benchmark(ctx, steps, resolution, profile_out, flow, vtk_out, use_native):
    """Run a short simulation and print performance in MLUPS.
    """
    # start profiling
    if profile_out:
        profile = cProfile.Profile()
        profile.enable()

    # setup and run simulation
    device, dtype = ctx.obj['device'], ctx.obj['dtype']
    flow_class, stencil = flow_by_name[flow]
    lattice = Lattice(stencil, device, dtype, use_native=use_native)
    flow = flow_class(resolution=resolution, reynolds_number=1, mach_number=0.05, lattice=lattice)
    force = Guo(
        lattice,
        tau=flow.units.relaxation_parameter_lu,
        acceleration=flow.units.convert_acceleration_to_lu(flow.force)
    ) if hasattr(flow, "acceleration") else None
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu, force=force)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    if vtk_out:
        simulation.reporters.append(VTKReporter(lattice, flow, interval=10))
    mlups = simulation.step(num_steps=steps)

    # write profiling output
    if profile_out:
        profile.disable()
        stats = pstats.Stats(profile)
        stats.sort_stats('cumulative')
        stats.print_stats()
        profile.dump_stats(profile_out)
        click.echo(f"Saved profiling information to {profile_out}.")

    click.echo("Finished {} steps in {} bit precision. MLUPS: {:10.2f}".format(
        steps, str(dtype).replace("torch.float", ""), mlups))
    return 0


@main.command()
@click.option("--init_f_neq/--no-initfneq", default=False, help="Initialize fNeq via finite differences")
@click.option("--use-native/--use-no-native", default=True, help="whether to use the native implementation or not.")
@click.pass_context
def convergence(ctx, init_f_neq, use_native):
    """Use Taylor Green 2D for convergence test in diffusive scaling."""
    device, dtype = ctx.obj['device'], ctx.obj['dtype']
    lattice = Lattice(D2Q9, device, dtype, use_native=use_native)
    error_u_old = None
    error_p_old = None
    print(("{:>15} " * 5).format("resolution", "error (u)", "order (u)", "error (p)", "order (p)"))

    for i in range(4, 9):
        resolution = 2 ** i
        mach_number = 8 / resolution

        # Simulation
        flow = TaylorGreenVortex2D(resolution=resolution, reynolds_number=10000, mach_number=mach_number,
                                   lattice=lattice)
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
        streaming = StandardStreaming(lattice)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
        if init_f_neq:
            simulation.initialize_f_neq()
        error_reporter = ErrorReporter(lattice, flow, interval=1, out=None)
        simulation.reporters.append(error_reporter)
        for _ in range(10 * resolution):
            simulation.step(1)
        error_u, error_p = np.mean(np.abs(error_reporter.out), axis=0).tolist()
        factor_u = 0 if error_u_old is None else error_u_old / error_u
        factor_p = 0 if error_p_old is None else error_p_old / error_p
        error_u_old = error_u
        error_p_old = error_p
        print("{:15} {:15.2e} {:15.1f} {:15.2e} {:15.1f}".format(
            resolution, error_u, factor_u / 2, error_p, factor_p / 2))
    if factor_u / 2 < 1.9:
        print("Velocity convergence order < 2.")
    if factor_p / 2 < 0.9:
        print("Velocity convergence order < 1.")
    if factor_u / 2 < 1.9 or factor_p / 2 < 0.9:
        sys.exit(1)
    else:
        return 0


from lettuce import Boundary
from lettuce import NativeBounceBackBoundary


class BounceBackBoundary(Boundary):
    """Fullway Bounce-Back Boundary"""

    def native_available(self) -> bool:
        return True

    def create_native(self) -> ['NativeLatticeBase']:
        return [NativeBounceBackBoundary.create()]

    def __init__(self, mask, lattice):
        Boundary.__init__(self, lattice)
        self.no_boundary_mask = lattice.convert_to_tensor(mask)

    def __call__(self, f):
        f = torch.where(self.no_boundary_mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.no_boundary_mask.shape == f_shape[1:]
        return self.no_boundary_mask


class Obstacle:
    def __init__(self, shape, reynolds_number, mach_number, lattice, domain_length_x, char_length=1, char_velocity=1):
        self.shape = shape
        char_length_lu = shape[0] / domain_length_x * char_length
        self.units = lettuce.UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length,
            characteristic_velocity_pu=char_velocity
        )
        self._mask = np.zeros(shape=self.shape, dtype=bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.shape
        self._mask = m.astype(bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u = np.array(2 * [(1 - self.mask) * self.units.characteristic_velocity_pu])
        u[0, 2, 1:3] = u[0, 2, 1:3] * np.sin(x[1][2, 1:3]) / 2 + 1
        u[1] *= 0
        return p, u

    @property
    def grid(self):
        xy = tuple(self.units.convert_length_to_pu(np.arange(n)) for n in self.shape)
        return np.meshgrid(*xy, indexing='ij')

    @property
    def boundaries(self):
        return [BounceBackBoundary(self.mask, self.units.lattice)]


@main.command()
@click.option("--use-native/--use-no-native", default=True, help="whether to use the native implementation or not.")
@click.pass_context
def pipetest(ctx, use_native):
    device, dtype = ctx.obj['device'], ctx.obj['dtype']

    lattice = lettuce.Lattice(lettuce.D2Q9, device, dtype, use_native=use_native)
    # flow = lettuce.TaylorGreenVortex3D(resolution=64, reynolds_number=10, mach_number=0.05, lattice=lattice)

    shape = (128, 64)
    resolution = shape[0] * shape[1]
    flow = Obstacle(shape=(128, 64), reynolds_number=100, mach_number=0.1, lattice=lattice, domain_length_x=10.1)
    x, y = flow.grid
    condition = np.sqrt((x - 2.5) ** 2 + (y - 2.5) ** 2) < 0.25
    flow.mask[np.where(condition)] = 1

    boundaries = flow.boundaries
    collision = lettuce.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lettuce.StandardStreaming(lattice)
    simulation = lettuce.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

    step_count = 1000
    first_step = lambda step: step == 0
    last_step = lambda step: step == step_count - 1
    inbetween_steps = lambda step: not first_step(step) and not last_step(step)

    pipeline_steps = [

        (Read(lattice), first_step),
        (collision, first_step),
        *[(boundary, first_step) for boundary in boundaries],
        (Write(lattice), first_step),

        (StandardRead(lattice), inbetween_steps),
        (collision, inbetween_steps),
        *[(boundary, inbetween_steps) for boundary in boundaries],
        (Write(lattice), inbetween_steps),

        (StandardRead(lattice), last_step),
        (Write(lattice), last_step)
    ]
    pipeline1 = Pipeline(lattice, pipeline_steps)

    start = timer()
    for i in range(step_count):
        pipeline1(simulation)
    end = timer()

    u = lattice.u(simulation.f)
    plt.imshow(lattice.convert_to_numpy(torch.norm(u, dim=0).cpu()))
    plt.show()

    print("Performance in MLUPS:", ((resolution ** 2) * step_count) / ((end - start) * 1000000))


if __name__ == "__main__":
    sys.exit(main())
