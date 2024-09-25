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

from lettuce import *
from lettuce import __version__ as lettuce_version
from lettuce.ext import (BGKCollision, ErrorReporter, VTKReporter,
                         flow_by_name, Guo)


@click.group()
@click.version_option(version=lettuce_version)
@click.option("--cuda/--no-cuda",
              default=True,
              help="Use cuda (default=True).")
@click.option("-i", "--gpu-id",
              type=int,
              default=0,
              help="Device ID of the GPU (default=0).")
@click.option("-p", "--precision",
              type=click.Choice(["half", "single", "double"]),
              default="double",
              help="Numerical Precision; 16, 32, or 64 bit per float "
                   "(default=double).")
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
    dtype = {"half": torch.half, "single": torch.float,
             "double": torch.double}[precision]

    ctx.obj['device'] = device
    ctx.obj['dtype'] = dtype


@main.command()
@click.option("-s", "--steps",
              type=int,
              default=10,
              help="Number of time steps.")
@click.option("-r", "--resolution",
              type=int,
              default=1024,
              help="Grid Resolution")
@click.option("-o", "--profile-out",
              type=str,
              default="",
              help="File to write profiling information to (default=""; "
                   "no profiling information gets written).")
@click.option("-f", "--flow",
              type=click.Choice(list(flow_by_name.keys())),
              default="taylor2d")
@click.option("-v", "--vtk-out",
              type=str,
              default="",
              help="VTK file basename to write the velocities and densities "
                   "to (default=""; no info gets written).")
@click.option("--use-cuda_native/--use-no-cuda_native",
              default=True,
              help="whether to use the cuda_native implementation or not.")
@click.pass_context  # pass parameters to sub-commands
def benchmark(ctx, steps, resolution, profile_out, flow, vtk_out,
              use_cuda_native):
    """Run a short simulation and print performance in MLUPS.
    """
    # start profiling
    if profile_out:
        profile = cProfile.Profile()
        profile.enable()

    # setup and run simulation

    flow_class, stencil = flow_by_name[flow]
    context = Context(ctx.obj['device'], ctx.obj['dtype'], use_cuda_native)

    flow = flow_class(context, resolution=resolution, reynolds_number=1,
                      mach_number=0.05, stencil=stencil)

    force = Guo(
        tau=flow.units.relaxation_parameter_lu,
        acceleration=flow.units.convert_acceleration_to_lu(flow.force)
    ) if hasattr(flow, "acceleration") else None

    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu,
                             force=force)
    reporter = []
    if vtk_out:
        reporter.append(VTKReporter(interval=10))

    simulation = Simulation(flow, collision, reporter)
    mlups = simulation(num_steps=steps)

    # write profiling output
    if profile_out:
        profile.disable()
        stats = pstats.Stats(profile)
        stats.sort_stats('cumulative')
        stats.print_stats()
        profile.dump_stats(profile_out)
        click.echo(f"Saved profiling information to {profile_out}.")

    click.echo("Finished {} steps in {} bit precision. MLUPS: {:10.2f}".format(
        steps, str(ctx.obj['dtype']).replace("torch.float", ""), mlups))
    return 0


@main.command()
@click.option("--use-cuda_native/--use-no-cuda_native",
              default=True,
              help="whether to use the cuda_native implementation or not.")
@click.pass_context
def convergence(ctx, use_cuda_native):
    """Use Taylor Green 2D for convergence test in diffusive scaling."""
    use_cuda_native &= ctx.obj['device'].type != 'cpu'
    context = Context(ctx.obj['device'], ctx.obj['dtype'],
                      use_native=use_cuda_native)

    error_u_old = None
    error_p_old = None
    factor_u = None
    factor_p = None
    print(("{:>15} " * 6).format("resolution", "error (u)", "order (u)",
                                 "error (p)", "order (p)", "MLUPS"))

    for i in range(4, 9):
        resolution = 2 ** i
        mach_number = 8 / resolution

        # Simulation
        flow = TaylorGreenVortex(context, [resolution] * 2,
                                 reynolds_number=10000,
                                 mach_number=mach_number)
        collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
        error_reporter = ErrorReporter(flow.analytic_solution, interval=1,
                                       out=None)

        simulation = Simulation(flow, collision, [error_reporter])

        mlups = simulation(10 * resolution)

        error_u, error_p = np.mean(np.abs(error_reporter.out), axis=0).tolist()
        factor_u = 0 if error_u_old is None else error_u_old / error_u
        factor_p = 0 if error_p_old is None else error_p_old / error_p
        error_u_old = error_u
        error_p_old = error_p

        print(f"{resolution:15} {error_u:15.2e} {factor_u / 2:15.2f} "
              f"{error_p:15.2e} {factor_p / 2:15.2f} {mlups:15.2f}")
    tol = 1e-1
    if not (2 - tol) < factor_u / 2 < (2 + tol):
        print(f"FAILED: Velocity convergence order {factor_u / 2} is not in "
              f"[1.9, 2.1")
        sys.exit(1)
    if not (1 - tol) < factor_p / 2 < (1 + tol):
        print(f"FAILED: Pressure convergence order {factor_p / 2} is not in "
              f"[0.9, 1.1].")
        sys.exit(1)
    else:
        return 0


if __name__ == "__main__":
    # convergence([], use_native=False)
    sys.exit(main(['--cuda', '-p', 'single', 'benchmark', '--steps', '100',
                   '--resolution', '2048', '--use-no-cuda_native']))
