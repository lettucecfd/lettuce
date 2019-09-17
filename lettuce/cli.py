# -*- coding: utf-8 -*-

"""Console script for lettuce."""

import sys
import cProfile
import pstats

import click
import torch
import numpy as np

import lettuce
from lettuce import BGKCollision, StandardStreaming, Lattice, LatticeAoS, D2Q9
from lettuce import TaylorGreenVortex2D, Simulation, ErrorReporter
from lettuce.flows import channel, couette
from lettuce.boundary import BounceBackBoundary

@click.group()
@click.version_option(version=lettuce.__version__)
@click.option("--cuda/--no-cuda", default=True, help="Use cuda (default=True).")
@click.option("-field", "--gpu-id", type=int, default=0, help="Device ID of the GPU (default=0).")
@click.option("-p", "--precision", type=click.Choice(["half", "single", "double"]), default="single",
              help="Numerical Precision; 16, 32, or 64 bit per float (default=single).")
@click.option("--aos/--no-aos", default=False, help="Use array-of-structure data storage order.")
@click.pass_context  # pass parameters to sub-commands
def main(ctx, cuda, gpu_id, precision, aos):
    """Pytorch-accelerated Lattice Boltzmann Solver
    """
    ctx.obj = {'device': None, 'dtype': None, 'aos': None}
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
    ctx.obj['aos'] = aos


@main.command()
@click.option("-s", "--steps", type=int, default=10, help="Number of time steps.")
@click.option("-r", "--resolution", type=int, default=1024, help="Grid Resolution")
@click.option("-o", "--profile-out", type=str, default="",
              help="File to write profiling information to (default=""; no profiling information gets written).")
@click.pass_context  # pass parameters to sub-commands
def benchmark(ctx, steps, resolution, profile_out):
    """Run a short simulation and print performance in MLUPS.
    """
    # start profiling
    profile = cProfile.Profile()
    profile.enable()

    # setup and run simulation
    device, dtype, aos = ctx.obj['device'], ctx.obj['dtype'], ctx.obj['aos']
    if aos:
        lattice = LatticeAoS(D2Q9, device, dtype)
    else:
        lattice = Lattice(D2Q9, device, dtype)
    flow = TaylorGreenVortex2D(resolution=resolution, reynolds_number=1, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming)
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
@click.option("-s", "--steps", type=int, default=300, help="Number of time steps.")
@click.option("-r", "--resolution", type=int, default=200, help="Grid Resolution")
@click.option("-o", "--profile-out", type=str, default="",
              help="File to write profiling information to (default=""; no profiling information gets written).")
@click.pass_context  # pass parameters to sub-commands
def channelflow(ctx, steps, resolution, profile_out):
    """Run a short simulation and print performance in MLUPS.
    """
    # start profiling
    profile = cProfile.Profile()
    profile.enable()

    # setup and run simulation
    device, dtype, aos = ctx.obj['device'], ctx.obj['dtype'], ctx.obj['aos']
    if aos:
        lattice = LatticeAoS(D2Q9, device, dtype)
    else:
        lattice = Lattice(D2Q9, device, dtype)
    flow = channel.ChannelFlow2D(resolution=resolution, reynolds_number=1, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    a = np.zeros((resolution, resolution*2), dtype=bool)
    a[:, 1] = True
    a[:, -1] = True
    a[1, :] = True
    a[-1, :] = True
    #a = bytes(a.any())
    boundary = BounceBackBoundary(mask=a, lattice=lattice)
    simulation = Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming, boundary=boundary)
    mlups = simulation.step(num_steps=steps)

#    x = flow.grid()
 #   nx = x[:,1]
    #print(simulation.p)

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
    device, dtype, aos = ctx.obj['device'], ctx.obj['dtype'], ctx.obj['aos']
    if aos:
        lattice = LatticeAoS(D2Q9, device, dtype)
    else:
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
            simulation.step(1)#num_steps=10*resolution)
            #print(flow.units.convert_density_lu_to_pressure_pu(lattice.rho(simulation.f))[0,0,0])

        # error calculation
        #print(error_reporter.out)
        error_u, error_p = np.mean(np.abs(error_reporter.out), axis=0).tolist()
        factor_u = 0 if error_u_old is None else error_u_old / error_u
        factor_p = 0 if error_p_old is None else error_p_old / error_p
        error_u_old = error_u
        error_p_old = error_p
        print("{:15} {:15.2e} {:15.1f} {:15.2e} {:15.1f}".format(
            resolution, error_u, factor_u/2, error_p, factor_p/2))
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

