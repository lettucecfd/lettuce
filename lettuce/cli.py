# -*- coding: utf-8 -*-

"""Console script for lettuce.
To get help for terminal commands, open a console and type:

>>>  lettuce --help

"""

import sys
import cProfile
import pstats

import click
import torch
import numpy as np

import lettuce
from lettuce import BGKCollision, StandardStreaming, Lattice, D2Q9

from lettuce import TaylorGreenVortex2D, Simulation, ErrorReporter, VTKReporter
from lettuce.flows import flow_by_name
from lettuce.force import Guo


@click.group()
@click.version_option(version=lettuce.__version__)
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
@click.pass_context  # pass parameters to sub-commands
def benchmark(ctx, steps, resolution, profile_out, flow, vtk_out):
    """Run a short simulation and print performance in MLUPS.
    """
    # start profiling
    profile = cProfile.Profile()
    profile.enable()

    # setup and run simulation
    device, dtype = ctx.obj['device'], ctx.obj['dtype']
    lattice = Lattice(D2Q9, device, dtype)
    flow_class = flow_by_name[flow]
    flow = flow_class(resolution=resolution, reynolds_number=1, mach_number=0.05, lattice=lattice)
    force = Guo(
        lattice,
        tau=flow.units.relaxation_parameter_lu,
        acceleration=flow.units.convert_acceleration_to_lu(flow.force)
    ) if hasattr(flow, "acceleration") else None
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu, force=force)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming)
    if vtk_out:
        simulation.reporters.append(VTKReporter(lattice, flow, interval=10))
    mlups = simulation.step(num_steps=steps)

    # write profiling output
    profile.disable()
    if profile_out:
        stats = pstats.Stats(profile)
        stats.sort_stats('cumulative')
        stats.print_stats()
        profile.dump_stats(profile_out)
        click.echo(f"Saved profiling information to {profile_out}.")

    click.echo("Finished {} steps in {} bit precision. MLUPS: {:10.2f}".format(
        steps, str(dtype).replace("torch.float",""), mlups))
    return 0


@main.command()
@click.pass_context
def convergence(ctx):
    """Use Taylor Green 2D for convergence test in diffusive scaling."""
    device, dtype = ctx.obj['device'], ctx.obj['dtype']
    lattice = Lattice(D2Q9, device, dtype)
    error_u_old = None
    error_p_old = None
    print(("{:>15} " * 5).format("resolution", "error (u)", "order (u)", "error (p)", "order (p)"))

    for i in range(4,9):
        resolution = 2**i
        mach_number = 8/resolution

        # Simulation
        flow = TaylorGreenVortex2D(resolution=resolution, reynolds_number=10000, mach_number=mach_number, lattice=lattice)
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
        streaming = StandardStreaming(lattice)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
        error_reporter = ErrorReporter(lattice, flow, interval=1, out=None)
        simulation.reporters.append(error_reporter)
        for i in range(10*resolution):
            simulation.step(1)
        error_u, error_p = np.mean(np.abs(error_reporter.out), axis=0).tolist()
        factor_u = 0 if error_u_old is None else error_u_old / error_u
        factor_p = 0 if error_p_old is None else error_p_old / error_p
        error_u_old = error_u
        error_p_old = error_p
        print("{:15} {:15.2e} {:15.1f} {:15.2e} {:15.1f}".format(
            resolution, error_u, factor_u/2, error_p, factor_p/2))
    if factor_u/2 < 1.9:
        print("Velocity convergence order < 2.")
    if factor_p/2 < 0.9:
        print("Velocity convergence order < 1.")
    if factor_u / 2 < 1.9 or factor_p/2 < 0.9:
        sys.exit(1)
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
