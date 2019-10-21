import pytest
import torch
from lettuce import D2Q9, VTKReporter
from lettuce import Lattice, Simulation, BGKCollision, StandardStreaming
from lettuce.flows.poiseuille import PoiseuilleFlow2D
from lettuce.force import Guo, ShanChen

def test_force_guo():
    dtype = torch.double
    device = "cpu"
    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = PoiseuilleFlow2D(resolution=51, reynolds_number=17, mach_number=0.05, lattice=lattice)
    [p_ref, u_ref] = flow.initial_solution(flow.grid)
    u_ref = flow.units.convert_velocity_to_lu(u_ref)
    force = Guo(lattice, tau=flow.units.relaxation_parameter_lu, F=flow.F)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu, force=force)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(100)
    u_sim = lattice.convert_to_numpy(lattice.u(simulation.f))
    assert u_ref[0].max() == pytest.approx(u_sim[0].max(), 1e-3)

def test_force_shanchen():
    dtype = torch.double
    device = "cpu"
    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = PoiseuilleFlow2D(resolution=51, reynolds_number=17, mach_number=0.05, lattice=lattice)
    [p_ref, u_ref] = flow.initial_solution(flow.grid)
    u_ref = flow.units.convert_velocity_to_lu(u_ref)
    force = ShanChen(lattice, tau=flow.units.relaxation_parameter_lu, F=flow.F)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu, force=force)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(100)
    u_sim = lattice.convert_to_numpy(lattice.u(simulation.f))
    assert u_ref[0].max() == pytest.approx(u_sim[0].max(), 1e-3)