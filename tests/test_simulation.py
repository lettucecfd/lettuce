"""Tests for simulation"""

import pytest
import numpy as np
from lettuce import Simulation, TaylorGreenVortex2D, Lattice, D2Q9, BGKCollision, StandardStreaming


# Note: Simulation is also implicitly tested in test_flows


def test_save_and_load(dtype_device, tmpdir):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype)
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(10)
    simulation.save_checkpoint(tmpdir/"checkpoint.pic")
    simulation2 = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation2.load_checkpoint(tmpdir/"checkpoint.pic")
    assert lattice.convert_to_numpy(simulation2.f) == pytest.approx(lattice.convert_to_numpy(simulation.f))


def test_initialization(dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype)
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    # set initial pressure to 0 everywhere
    p, u = flow.initial_solution(flow.grid)
    u0 = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
    rho0 = lattice.convert_to_tensor(np.ones_like(u0[[0],...]))
    simulation.f = lattice.equilibrium(rho0, u0)
    num_iterations = simulation.initialize(500, 0.001)
    piter = lattice.convert_to_numpy(flow.units.convert_density_lu_to_pressure_pu(lattice.rho(simulation.f)))
    # assert that pressure is converged up to 0.05 (max p
    assert piter == pytest.approx(p, abs=5e-2)
    assert num_iterations < 500
