import pytest
import torch
import numpy as np
from lettuce import D2Q9
from lettuce import Lattice, Simulation, BGKCollision, StandardStreaming
from lettuce.flows.poiseuille import PoiseuilleFlow2D
from lettuce.force import Guo, ShanChen


@pytest.mark.parametrize("ForceType", [Guo, ShanChen])
def test_force(ForceType, device):
    dtype = torch.double
    lattice = Lattice(D2Q9, dtype=dtype, device=device, use_native=False)  # TODO use_native Fails here
    flow = PoiseuilleFlow2D(resolution=16, reynolds_number=1, mach_number=0.02, lattice=lattice,
                            initialize_with_zeros=True)
    acceleration_lu = flow.units.convert_acceleration_to_lu(flow.acceleration)
    force = ForceType(lattice, tau=flow.units.relaxation_parameter_lu,
                      acceleration=acceleration_lu)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu, force=force)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(1000)
    # compare with reference solution
    u_sim = lattice.u(simulation.f, acceleration=torch.as_tensor(acceleration_lu, device=device))
    u_sim = flow.units.convert_velocity_to_pu(lattice.convert_to_numpy(u_sim))
    _, u_ref = flow.analytic_solution(flow.grid)
    fluidnodes = np.where(np.logical_not(flow.boundaries[0].mask.cpu()))
    assert u_ref[0].max() == pytest.approx(u_sim[0].max(), rel=0.01)
    assert u_ref[0][fluidnodes] == pytest.approx(u_sim[0][fluidnodes], rel=None, abs=0.01 * u_ref[0].max())
