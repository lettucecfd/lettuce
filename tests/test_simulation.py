"""Tests for simulation"""

import pytest
from lettuce import Simulation, TaylorGreenVortex2D, Lattice, LatticeAoS, D2Q9, BGKCollision, StandardStreaming


# Note: Simulation is implicitly tested in test_flows


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


def test_aos_equivalent(dtype_device):
    dtype, device = dtype_device
    result = dict()
    for L in [LatticeAoS, Lattice]:
        lattice = L(D2Q9, device, dtype)
        flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
        streaming = StandardStreaming(lattice)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
        simulation.step(10)
        result[L] = lattice.convert_to_numpy(simulation.f)
    assert result[Lattice] == pytest.approx(result[LatticeAoS])
