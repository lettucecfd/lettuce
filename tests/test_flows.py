
import pytest
import numpy as np
import torch
from lettuce import TaylorGreenVortex2D, TaylorGreenVortex3D, CouetteFlow2D, D2Q9, D3Q27, DoublyPeriodicShear2D
from lettuce import Lattice, Simulation, BGKCollision, BGKInitialization, StandardStreaming
from lettuce.flows.poiseuille import PoiseuilleFlow2D

# Flows to test
INCOMPRESSIBLE_2D = [TaylorGreenVortex2D, CouetteFlow2D, PoiseuilleFlow2D, DoublyPeriodicShear2D]
INCOMPRESSIBLE_3D = [TaylorGreenVortex3D]


@pytest.mark.parametrize("IncompressibleFlow", INCOMPRESSIBLE_2D)
def test_flow_2d(IncompressibleFlow, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = IncompressibleFlow(16, 1, 0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(1)


@pytest.mark.parametrize("IncompressibleFlow", INCOMPRESSIBLE_3D)
def test_flow_3d(IncompressibleFlow, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D3Q27, dtype=dtype, device=device)
    flow = IncompressibleFlow(16, 1, 0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(1)


