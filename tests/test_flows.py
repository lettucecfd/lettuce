import pytest
import numpy as np
import torch
from lettuce import TaylorGreenVortex2D, TaylorGreenVortex3D, CouetteFlow2D, D2Q9, D3Q27, DoublyPeriodicShear2D
from lettuce import torch_gradient, DecayingTurbulence
from lettuce import Lattice, Simulation, BGKCollision, BGKInitialization, StandardStreaming
from lettuce import Obstacle2D, Obstacle3D
from lettuce.flows.poiseuille import PoiseuilleFlow2D

# Flows to test
INCOMPRESSIBLE_2D = [TaylorGreenVortex2D, CouetteFlow2D, PoiseuilleFlow2D, DoublyPeriodicShear2D, DecayingTurbulence]
INCOMPRESSIBLE_3D = [TaylorGreenVortex3D, DecayingTurbulence]


@pytest.mark.parametrize("IncompressibleFlow", INCOMPRESSIBLE_2D)
def test_flow_2d(IncompressibleFlow, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device, use_native=False)
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


@pytest.mark.parametrize("stencil", [D2Q9, D3Q27])
def test_divergence(stencil, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(stencil, dtype=dtype, device=device)
    flow = DecayingTurbulence(50, 1, 0.05, lattice=lattice, ic_energy=0.5)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    ekin = flow.units.convert_incompressible_energy_to_pu(
        torch.sum(lattice.incompressible_energy(simulation.f))) * flow.units.convert_length_to_pu(1.0) ** lattice.D

    u0 = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)[0])
    u1 = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)[1])
    dx = flow.units.convert_length_to_pu(1.0)
    grad_u0 = torch_gradient(u0, dx=dx, order=6).cpu().numpy()
    grad_u1 = torch_gradient(u1, dx=dx, order=6).cpu().numpy()
    divergence = np.sum(grad_u0[0] + grad_u1[1])

    if lattice.D == 3:
        u2 = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)[2])
        grad_u2 = torch_gradient(u2, dx=dx, order=6).cpu().numpy()
        divergence += np.sum(grad_u2[2])
    assert (flow.ic_energy == pytest.approx(lattice.convert_to_numpy(ekin), rel=1))
    assert (0 == pytest.approx(divergence, abs=2e-3))


@pytest.mark.parametrize("stencil", [D2Q9, D3Q27])
def test_obstacle(stencil, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(stencil, dtype=dtype, device=device)

    nx = 32
    ny = 16
    nz = 16

    if stencil is D2Q9:
        mask = np.zeros([nx, ny])
        mask[3:6, 3:6] = 1
        flow = Obstacle2D(nx, ny, 100, 0.1, lattice=lattice, char_length_lu=3)
    if stencil is D3Q27:
        mask = np.zeros([nx, ny, nz])
        mask[3:6, 3:6, :] = 1
        flow = Obstacle3D(nx, ny, nz, 100, 0.1, lattice=lattice, char_length_lu=3)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    flow.mask = mask != 0
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    simulation.step(2)
