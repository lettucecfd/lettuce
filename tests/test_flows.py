
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


def test_pressure_initialization_tgv2d(device):
    pytest.skip("Not finished, yet.")
    lattice = Lattice(D2Q9, dtype=torch.double, device=device)
    flow = TaylorGreenVortex2D(16, 0.1, 0.1, lattice=lattice)
    from lettuce.moments import D2Q9Lallemand
    moment_trafo = D2Q9Lallemand(lattice)
    collision = BGKInitialization(lattice, flow=flow, moment_transformation=moment_trafo)
    streaming = StandardStreaming(lattice)
    p, u = flow.initial_solution(flow.grid)
    u_lu = flow.units.convert_velocity_to_lu(lattice.convert_to_tensor(u))
    # intialize with unit density and see if we arrive at the right pressure
    rho_lu = flow.units.convert_pressure_pu_to_density_lu(lattice.convert_to_tensor(p))
    #f = lattice.equilibrium(rho_lu, u_lu)
    f = lattice.equilibrium(0, u_lu)
    for i in range(200):
        f = streaming(f)
        f = collision(f)
        p_num = lattice.convert_to_numpy(flow.units.convert_density_lu_to_pressure_pu(lattice.rho(f)+1))
        print(p_num[0,0,:5])
    #assert lattice.convert_to_numpy(lattice.u(f)) == pytest.approx(lattice.convert_to_numpy(u_lu))
    p_num = lattice.convert_to_numpy(flow.units.convert_density_lu_to_pressure_pu(lattice.rho(f)+1))
    print (np.max(np.abs((p-p_num))), np.max(np.abs(p)), np.max(np.abs(p_num)), np.abs(p[0,0,0]), np.abs(p_num[0,0,0]))
    print(p[0,0,:10]);print(p_num[0,0,:10])


