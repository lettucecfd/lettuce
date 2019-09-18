
import pytest
import numpy as np
import torch
from lettuce import TaylorGreenVortex2D, TaylorGreenVortex3D, CouetteFlow2D, D2Q9, D3Q27
from lettuce import Lattice, LatticeAoS, Simulation, BGKCollision, BGKInitialization, StandardStreaming

# Flows to test
INCOMPRESSIBLE_2D = [TaylorGreenVortex2D, CouetteFlow2D]
INCOMPRESSIBLE_3D = [TaylorGreenVortex3D]


@pytest.mark.parametrize("IncompressibleFlow", INCOMPRESSIBLE_2D)
@pytest.mark.parametrize("Ltc", [Lattice, LatticeAoS])
def test_flow_2d(IncompressibleFlow, Ltc, dtype_device):
    dtype, device = dtype_device
    lattice = Ltc(D2Q9, dtype=dtype, device=device)
    flow = IncompressibleFlow(16, 1, 0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(1)


@pytest.mark.parametrize("IncompressibleFlow", INCOMPRESSIBLE_3D)
@pytest.mark.parametrize("Ltc", [Lattice, LatticeAoS])
def test_flow_3d(IncompressibleFlow, Ltc, dtype_device):
    dtype, device = dtype_device
    lattice = Ltc(D3Q27, dtype=dtype, device=device)
    flow = IncompressibleFlow(16, 1, 0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(1)


