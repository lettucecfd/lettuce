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
    def bench(num_steps, resolution):
        print(f"bench for num_steps:{num_steps} and resolution:{resolution}")
        print(f"extension: {extension(num_steps, resolution)} mlups")
        print(f"baseline:  {baseline(num_steps, resolution)} mlups")
        print()


    bench(10, 1024)
    bench(100, 1024)
    bench(10, 3072)
    bench(100, 3072)
    bench(250, 2048)
