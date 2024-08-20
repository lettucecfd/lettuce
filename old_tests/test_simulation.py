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


@pytest.mark.parametrize("Case", [TaylorGreenVortex2D, TaylorGreenVortex3D,
                                  DecayingTurbulence])
def test_initialize_fneq(Case, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype, use_native=False)
    if "3D" in Case.__name__:
        lattice = Lattice(D3Q27, dtype=dtype, device=device, use_native=False)
    flow = Case(resolution=32, reynolds_number=1000, mach_number=0.1,
                lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation_neq = Simulation(flow, collision, [])

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
        error_reporter_neq = ErrorReporter(flow.analytic_solution, interval=1,
                                           out=None)
        error_reporter_eq = ErrorReporter(flow.analytic_solution,
                                          interval=1, out=None)
        simulation_eq = Simulation(flow, collision, [])
        simulation_neq.reporter.append(error_reporter_neq)
        simulation_eq.reporter.append(error_reporter_eq)

        simulation_neq(10)
        simulation_eq(10)
        error_u, error_p = np.mean(np.abs(error_reporter_neq.out),
                                   axis=0).tolist()
        error_u_eq, error_p_eq = np.mean(np.abs(error_reporter_eq.out),
                                         axis=0).tolist()

        assert error_u < error_u_eq
