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
from torch import nn

from lettuce import BGKCollision, StandardStreaming, Lattice, D2Q9
from lettuce import __version__ as lettuce_version

from lettuce import TaylorGreenVortex2D, Simulation, ErrorReporter, VTKReporter
from lettuce.flows import flow_by_name
from lettuce.force import Guo

from lettuce.extension import stream_and_collide
from copy import deepcopy
from timeit import default_timer as timer


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
@click.pass_context  # pass parameters to sub-commands
def benchmark(ctx, steps, resolution, profile_out, flow, vtk_out):
    """Run a short simulation and print performance in MLUPS.
    """
    # start profiling
    if profile_out:
        profile = cProfile.Profile()
        profile.enable()

    # setup and run simulation
    device, dtype = ctx.obj['device'], ctx.obj['dtype']
    flow_class, stencil = flow_by_name[flow]
    lattice = Lattice(stencil, device, dtype)
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
@click.option("--native", default=False, help="")
@click.pass_context
def convergence(ctx, init_f_neq, native):
    """Use Taylor Green 2D for convergence test in diffusive scaling."""
    device, dtype = ctx.obj['device'], ctx.obj['dtype']
    lattice = Lattice(D2Q9, device, dtype)
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

        f_next = None
        if native:
            f_next = deepcopy(simulation.f)
            f_next = f_next.to(device)

        for _ in range(10 * resolution):

            if native:
                if simulation.i == 0:
                    simulation._report()

                simulation.i += 1
                stream_and_collide(simulation.f, f_next, collision.tau)
                simulation.f, f_next = f_next, simulation.f

                for boundary in simulation._boundaries:
                    simulation.f = boundary(simulation.f)

                simulation._report()

            else:
                if simulation.i == 0:
                    simulation._report()

                simulation.i += 1

                simulation.f = simulation.streaming(simulation.f)
                simulation.f = simulation.collision(simulation.f)

                for boundary in simulation._boundaries:
                    simulation.f = boundary(simulation.f)

                simulation._report()

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


@main.command()
@click.option("--native", default=True, help="")
@click.pass_context
def scratch(ctx, native):
    device = torch.device("cuda:0")
    dtype = torch.double
    flow_class = TaylorGreenVortex2D
    stencil = D2Q9

    def init(resolution):
        lattice = Lattice(stencil, device, dtype)
        flow = flow_class(resolution=resolution, reynolds_number=1, mach_number=0.05, lattice=lattice)
        force = Guo(
            lattice,
            tau=flow.units.relaxation_parameter_lu,
            acceleration=flow.units.convert_acceleration_to_lu(flow.force)
        ) if hasattr(flow, "acceleration") else None
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu, force=force)
        streaming = StandardStreaming(lattice)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
        return lattice, collision, streaming, simulation

    def _stream_and_collide(simulation, native_, f_next=None):
        if native_:
            assert f_next is not None
            stream_and_collide(simulation.f, f_next, simulation.collision.tau)
            simulation.f, f_next = f_next, simulation.f
        else:
            simulation.streaming(simulation.f)
            simulation.collision(simulation.f)

    def simulate(num_steps, resolution, native_):

        lattice, collision, streaming, simulation = init(resolution)
        f_next = None if not native_ else torch.empty(simulation.f.shape, device=device, dtype=dtype)

        start = timer()
        for _ in range(num_steps):
            simulation.i += 1
            _stream_and_collide(simulation, native_=native_, f_next=f_next)

        seconds = timer() - start
        mlups = num_steps * simulation.lattice.rho(simulation.f).numel() / 1e6 / seconds
        return simulation.f, mlups, seconds

    def bench(num_steps, resolution):
        print(f"bench for num_steps:{num_steps} and resolution:{resolution}")

        _, mlups, _ = simulate(num_steps, resolution, native_=True)
        del _
        print(f"extension: {mlups} mlups")

        _, mlups, _ = simulate(num_steps, resolution, native_=False)
        del _
        print(f"baseline:  {mlups} mlups\n")

    def assert_equal(num_steps, resolution):
        print(f"assert equal for num_steps:{num_steps} and resolution:{resolution}")

        f_ext, _, _ = simulate(num_steps, resolution, native_=True)
        f_org, _, _ = simulate(num_steps, resolution, native_=False)
        mse = nn.MSELoss(reduction='none')(f_ext, f_org)

        print(f"mse_max:  {torch.max(mse)}\n"
              f"mse_min:  {torch.min(mse)}\n"
              f"mse_sum:  {torch.sum(mse)}\n")

    def test_streaming():
        f = torch.empty((9, 16, 16), device=device, dtype=dtype)
        f_next = torch.ones((9, 16, 16), device=device, dtype=dtype)

        for q in range(9):
            i = 1.0
            for x in range(16):
                for y in range(16):
                    f[q, x, y] = i
                    i = i + 1.0

        stream_and_collide(f, f_next, 1.0)

        torch.set_printoptions(edgeitems=8)
        for q in range(9):
            print(f"q := {q}\n{f[q]}\n{f_next[q]}\n")

    # test_streaming()

    # assert_equal(1, 128, native=native)
    # assert_equal(2, 128, native=native)
    # assert_equal(16, 128, native=native)
    # assert_equal(32, 128, native=native)
    # assert_equal(100, 128, native=native)

    # bench(8000, 3072, native_=native)

    simulate(1000, 2048, native_=True)

    # bench(250, 1024)
    # bench(250, 2048)
    # bench(250, 3072)
    # bench(1000, 3072)
    # bench(2000, 2048)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
