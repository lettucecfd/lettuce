# -*- coding: utf-8 -*-

"""Console script for lettuce.
To get help for terminal commands, open a console and type:

>>>  lettuce --help

"""

import sys
import cProfile
import pstats

import click
import pytest
import torch
import numpy as np
from torch import nn

from lettuce import BGKCollision, StandardStreaming, Lattice, D2Q9
from lettuce import __version__ as lettuce_version

from lettuce import TaylorGreenVortex2D, Simulation, ErrorReporter, VTKReporter
from lettuce.flows import flow_by_name
from lettuce.force import Guo

import lettuce.extension as cuda_ext
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
                cuda_ext.stream_and_collide(simulation.f, f_next, collision.tau)
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
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu, force=None)
        streaming = StandardStreaming(lattice)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
        return lattice, collision, streaming, simulation

    def test_stream():

        w, h = 32, 32

        # f = torch.empty((9, w, h), device=device, dtype=dtype)
        # for q in range(9):
        #     for x in range(w):
        #         for y in range(h):
        #             f[q, x, y] = x * h + y

        f = torch.rand((9, w, h), device=device, dtype=dtype)

        f_ext_next = torch.zeros((9, w, h), device=device, dtype=dtype)
        cuda_ext.stream(f, f_ext_next)

        lattice = Lattice(stencil, device, dtype)
        f_org_next = deepcopy(f)
        for i in range(1, lattice.Q):
            shifts = tuple(lattice.stencil.e[i])
            dims = tuple(np.arange(lattice.D))
            f_org_next[i] = torch.roll(f_org_next[i], shifts=shifts, dims=dims)

        mse = nn.MSELoss(reduction='none')(f_ext_next, f_org_next)
        for q in range(9):

            mse_max = torch.max(mse[q]).cpu()
            mse_min = torch.min(mse[q]).cpu()
            mse_sum = torch.sum(mse[q]).cpu()

            print(f"q={q}\n"
                  f"mse_max:  {mse_max}\n"
                  f"mse_min:  {mse_min}\n"
                  f"mse_sum:  {mse_sum}\n")

            if mse_sum > 0.0:
                torch.set_printoptions(edgeitems=8)
                torch.set_printoptions(precision=64)
                f_ext_next = f_ext_next.cpu().numpy()
                f_org_next = f_org_next.cpu().numpy()
                print(f"q := {q}\n"
                      f"{f_ext_next[q]}\n"
                      f"{f_org_next[q]}\n"
                      f"{f_org_next[q] - f_ext_next[q]}\n")
                return

    def test_collide():

        w, h = 32, 32

        # f = torch.empty((9, w, h), device=device, dtype=dtype)
        # for q in range(9):
        #     for x in range(w):
        #         for y in range(h):
        #             f[q, x, y] = x * h + y

        f = torch.rand((9, w, h), device=device, dtype=dtype)

        lattice = Lattice(stencil, device, dtype)

        f_ext_next = torch.zeros((9, w, h), device=device, dtype=dtype)
        cuda_ext.collide(f, f_ext_next, 1.0)
        f_ext_next = torch.nan_to_num(f_ext_next)

        f_org_next = f - (f - lattice.equilibrium(lattice.rho(f), lattice.u(f)))
        f_org_next = torch.nan_to_num(f_org_next)

        mse = nn.MSELoss(reduction='none')(f_ext_next, f_org_next)
        for q in range(9):

            mse_max = torch.max(mse[q]).cpu()
            mse_min = torch.min(mse[q]).cpu()
            mse_sum = torch.sum(mse[q]).cpu()

            print(f"q={q}\n"
                  f"mse_max:  {mse_max}\n"
                  f"mse_min:  {mse_min}\n"
                  f"mse_sum:  {mse_sum}\n")

            if mse_sum > float(1.0e-25):
                torch.set_printoptions(edgeitems=8)
                torch.set_printoptions(precision=64)
                f_ext_next = f_ext_next.cpu().numpy()
                f_org_next = f_org_next.cpu().numpy()
                print(f"q := {q}\n"
                      f"{f_ext_next[q]}\n"
                      f"{f_org_next[q]}\n"
                      f"{f_org_next[q] - f_ext_next[q]}\n")
                return

    def test_stream_collide():

        w, h = 32, 32

        # f = torch.empty((9, w, h), device=device, dtype=dtype)
        # for q in range(9):
        #     for x in range(w):
        #         for y in range(h):
        #             f[q, x, y] = x * h + y

        f = torch.rand((9, w, h), device=device, dtype=dtype)

        lattice = Lattice(stencil, device, dtype)

        f_ext_next = torch.zeros((9, w, h), device=device, dtype=dtype)
        cuda_ext.stream(f, f_ext_next)
        cuda_ext.collide(f_ext_next, f_ext_next, 1.0)
        f_ext_next = torch.nan_to_num(f_ext_next)

        f_org_next = deepcopy(f)

        for i in range(1, lattice.Q):
            shifts = tuple(lattice.stencil.e[i])
            dims = tuple(np.arange(lattice.D))
            f_org_next[i] = torch.roll(f_org_next[i], shifts=shifts, dims=dims)

        f_org_next = f_org_next - (f_org_next - lattice.equilibrium(lattice.rho(f_org_next), lattice.u(f_org_next)))
        f_org_next = torch.nan_to_num(f_org_next)

        mse = nn.MSELoss(reduction='none')(f_ext_next, f_org_next)
        for q in range(9):

            mse_max = torch.max(mse[q]).cpu()
            mse_min = torch.min(mse[q]).cpu()
            mse_sum = torch.sum(mse[q]).cpu()

            print(f"q={q}\n"
                  f"mse_max:  {mse_max}\n"
                  f"mse_min:  {mse_min}\n"
                  f"mse_sum:  {mse_sum}\n")

            if mse_sum > float(1.0e-25):
                torch.set_printoptions(edgeitems=8)
                torch.set_printoptions(precision=64)
                f_ext_next = f_ext_next.cpu().numpy()
                f_org_next = f_org_next.cpu().numpy()
                print(f"q := {q}\n"
                      f"{f_ext_next[q]}\n"
                      f"{f_org_next[q]}\n"
                      f"{f_org_next[q] - f_ext_next[q]}\n")
                return

    def test_stream_and_collide(num_steps, resolution):

        w, h = resolution, resolution

        f = torch.rand((9, w, h), dtype=dtype).cpu()

        lattice = Lattice(stencil, device, dtype)

        f_ext = deepcopy(f).cuda()
        f_ext_next = torch.zeros((9, w, h), device=device, dtype=dtype)
        for _ in range(num_steps):
            cuda_ext.stream_and_collide(f_ext, f_ext_next, 1.0)
            f_ext, f_ext_next = f_ext_next, f_ext
        del f_ext_next
        f_ext = torch.nan_to_num(f_ext).cpu()

        f_org = deepcopy(f).cuda()
        for _ in range(num_steps):
            for i in range(1, lattice.Q):
                shifts = tuple(lattice.stencil.e[i])
                dims = tuple(np.arange(lattice.D))
                f_org[i] = torch.roll(f_org[i], shifts=shifts, dims=dims)
            f_org = f_org - (f_org - lattice.equilibrium(lattice.rho(f_org), lattice.u(f_org)))

        f_org = torch.nan_to_num(f_org).cpu()

        mse = nn.MSELoss(reduction='none')(f_ext, f_org).cpu()
        for q in range(9):

            mse_max = torch.max(mse[q])
            mse_min = torch.min(mse[q])
            mse_sum = torch.sum(mse[q])

            print(f"q={q}\n"
                  f"mse_max:  {mse_max}\n"
                  f"mse_min:  {mse_min}\n"
                  f"mse_sum:  {mse_sum}\n")

            if mse_sum > float(0.000_000_000_000_000_1):
                torch.set_printoptions(edgeitems=8)
                torch.set_printoptions(precision=64)
                f_ext = f_ext.numpy()
                f_org = f_org.numpy()
                print(f"q := {q}\n"
                      f"{f_ext[q]}\n"
                      f"{f_org[q]}\n"
                      f"{f_org[q] - f_ext[q]}\n")
                return

    def simulate(num_steps, resolution):

        lattice, collision, streaming, simulation = init(resolution)

        start = timer()
        for _ in range(num_steps):
            simulation.i += 1
            simulation.streaming(simulation.f)
            simulation.collision(simulation.f)

        seconds = timer() - start
        mlups = num_steps * simulation.lattice.rho(simulation.f).numel() / 1e6 / seconds
        return simulation.f, mlups, seconds

    def simulate_extension(num_steps, resolution):

        lattice, collision, streaming, simulation = init(resolution)
        f_next = torch.empty(simulation.f.shape, device=device, dtype=dtype)

        start = timer()
        for _ in range(num_steps):
            simulation.i += 1
            cuda_ext.stream_and_collide(simulation.f, f_next, simulation.collision.tau)
            simulation.f, f_next = f_next, simulation.f

        seconds = timer() - start
        mlups = num_steps * simulation.lattice.rho(simulation.f).numel() / 1e6 / seconds
        return simulation.f, mlups, seconds

    def bench(num_steps, resolution):
        print(f"bench for num_steps:{num_steps} and resolution:{resolution}")

        _, mlups, _ = simulate_extension(num_steps, resolution)
        del _
        print(f"extension: {mlups} mlups")

        _, mlups, _ = simulate(num_steps, resolution)
        del _
        print(f"baseline:  {mlups} mlups\n")

    test_stream()
    test_collide()
    test_stream_collide()
    test_stream_and_collide(100, 1024)

    bench(250, 1024)
    bench(250, 2048)
    bench(250, 3072)
    bench(1000, 3072)
    bench(2000, 2048)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
