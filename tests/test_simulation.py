"""Tests for simulation"""

import pytest
import numpy as np
from lettuce import (
    Simulation, TaylorGreenVortex2D, TaylorGreenVortex3D, Lattice,
    D2Q9, D3Q27, BGKCollision, StandardStreaming, ErrorReporter,
    DecayingTurbulence
)
import torch


# Note: Simulation is also implicitly tested in test_flows


def test_save_and_load(dtype_device, tmpdir):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype)
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    simulation.step(10)
    simulation.save_checkpoint(tmpdir / "checkpoint.pic")
    simulation2 = Simulation(flow, lattice, collision, streaming)
    simulation2.load_checkpoint(tmpdir / "checkpoint.pic")
    assert lattice.convert_to_numpy(simulation2.f) == pytest.approx(lattice.convert_to_numpy(simulation.f))


@pytest.mark.parametrize("use_jacobi", [True, False])
def test_initialization(dtype_device, use_jacobi):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype)
    flow = TaylorGreenVortex2D(resolution=24, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    # set initial pressure to 0 everywhere
    p, u = flow.initial_solution(flow.grid)
    u0 = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
    rho0 = lattice.convert_to_tensor(np.ones_like(u0[0, ...].cpu()))
    simulation.f = lattice.equilibrium(rho0, u0)
    if use_jacobi:
        simulation.initialize_pressure(1000, 1e-6)
        num_iterations = 0
    else:
        num_iterations = simulation.initialize(500, 1e-3)
    piter = lattice.convert_to_numpy(flow.units.convert_density_lu_to_pressure_pu(lattice.rho(simulation.f)))
    # assert that pressure is converged up to 0.05 (max p
    assert piter == pytest.approx(p, rel=0.0, abs=5e-2)
    assert num_iterations < 500


@pytest.mark.parametrize("Case", [TaylorGreenVortex2D, TaylorGreenVortex3D, DecayingTurbulence])
def test_initialize_fneq(Case, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype, use_native=False)  # TODO use_native Fails here
    if "3D" in Case.__name__:
        lattice = Lattice(D3Q27, dtype=dtype, device=device, use_native=False)  # TODO use_native Fails here
    flow = Case(resolution=32, reynolds_number=1000, mach_number=0.1, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation_neq = Simulation(flow, lattice, collision, streaming)

    pre_rho = lattice.rho(simulation_neq.f)
    pre_u = lattice.u(simulation_neq.f)
    pre_ke = lattice.incompressible_energy(simulation_neq.f)

    simulation_neq.initialize_f_neq()

    post_rho = lattice.rho(simulation_neq.f)
    post_u = lattice.u(simulation_neq.f)
    post_ke = lattice.incompressible_energy(simulation_neq.f)
    tol = 1e-6
    assert torch.allclose(pre_rho, post_rho, rtol=0.0, atol=tol)
    assert torch.allclose(pre_u, post_u, rtol=0.0, atol=tol)
    assert torch.allclose(pre_ke, post_ke, rtol=0.0, atol=tol)

    if Case is TaylorGreenVortex2D:
        error_reporter_neq = ErrorReporter(lattice, flow, interval=1, out=None)
        error_reporter_eq = ErrorReporter(lattice, flow, interval=1, out=None)
        simulation_eq = Simulation(flow, lattice, collision, streaming)
        simulation_neq.reporters.append(error_reporter_neq)
        simulation_eq.reporters.append(error_reporter_eq)

        simulation_neq.step(10)
        simulation_eq.step(10)
        error_u, error_p = np.mean(np.abs(error_reporter_neq.out), axis=0).tolist()
        error_u_eq, error_p_eq = np.mean(np.abs(error_reporter_eq.out), axis=0).tolist()

        assert error_u < error_u_eq
