from lettuce import TaylorGreenVortex2D, D2Q9, Lattice, Guo, BGKCollision, StandardStreaming, Simulation
from timeit import default_timer as timer
import torch
from copy import deepcopy

from lettuce.extension import stream_and_collide


def extension(num_steps, resolution):
    device = torch.device("cuda:0")
    dtype = torch.double
    flow_class = TaylorGreenVortex2D
    stencil = D2Q9

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

    # simulation (modified)

    f_next = deepcopy(simulation.f)

    def collision_operator(f, c):
        rho = c.lattice.rho(f)

        u_equilibrium = 0 if c.force is None else c.force.u_eq(f)
        u = c.lattice.u(f) + u_equilibrium

        f_equilibrium = c.lattice.equilibrium(rho, u)
        tau = c.tau

        return -((f - f_equilibrium) / tau)

    start = timer()
    for _ in range(num_steps):

        simulation.i += 1

        c = collision_operator(simulation.f, collision)  # TODO this will disappear later
        stream_and_collide(simulation.f, f_next, c)
        simulation.f, f_next = f_next, simulation.f

        for boundary in simulation._boundaries:
            simulation.f = boundary(simulation.f)

    end = timer()
    seconds = end - start
    num_grid_points = simulation.lattice.rho(simulation.f).numel()
    mlups = num_steps * num_grid_points / 1e6 / seconds

    return mlups


def baseline(num_steps, resolution):
    device = torch.device("cuda:0")
    dtype = torch.double
    flow_class = TaylorGreenVortex2D
    stencil = D2Q9

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

    # simulation (original)

    start = timer()
    for _ in range(num_steps):

        simulation.i += 1

        simulation.streaming(simulation.f)
        simulation.collision(simulation.f)

        for boundary in simulation._boundaries:
            simulation.f = boundary(simulation.f)

    end = timer()
    seconds = end - start
    num_grid_points = simulation.lattice.rho(simulation.f).numel()
    mlups = num_steps * num_grid_points / 1e6 / seconds

    return mlups


if __name__ == '__main__':

    print("bench for num_steps:10 and resolution:1026")
    print(f"extension: {extension(10, 1026)} mlups")
    print(f"baseline:  {baseline(10, 1026)} mlups")
    print()

    print("bench for num_steps:100 and resolution:1026")
    print(f"extension: {extension(100, 1026)} mlups")
    print(f"baseline:  {baseline(100, 1026)} mlups")
    print()

    print("bench for num_steps:10 and resolution:3074")
    print(f"extension: {extension(10, 3074)} mlups")
    print(f"baseline:  {baseline(10, 3074)} mlups")
    print()

    print("bench for num_steps:100 and resolution:3074")
    print(f"extension: {extension(100, 3074)} mlups")
    print(f"baseline:  {baseline(100, 3074)} mlups")
    print()

    print("bench for num_steps:250 and resolution:2050")
    print(f"extension: {extension(250, 2050)} mlups")
    print(f"baseline:  {baseline(250, 2050)} mlups")
