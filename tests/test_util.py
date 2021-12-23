import numpy as np
import torch
from lettuce import (
    Lattice, D2Q9, D3Q27, TaylorGreenVortex2D,
    TaylorGreenVortex3D, torch_gradient, grid_fine_to_coarse,
    BGKCollision, Simulation, StandardStreaming
)
from lettuce.util import pressure_poisson
import pytest


@pytest.mark.parametrize("order", [2, 4, 6])
def test_torch_gradient_2d(order):
    lattice = Lattice(D2Q9, device='cpu', )
    flow = TaylorGreenVortex2D(resolution=100, reynolds_number=1, mach_number=0.05, lattice=lattice)
    grid = flow.grid
    p, u = flow.initial_solution(grid)
    dx = flow.units.convert_length_to_pu(1.0)
    u0_grad = torch_gradient(lattice.convert_to_tensor(u[0]), dx=dx, order=order).numpy()
    u0_grad_np = np.array(np.gradient(u[0], dx))
    u0_grad_analytic = np.array([
        -np.sin(grid[0]) * np.sin(grid[1]),
        np.cos(grid[0]) * np.cos(grid[1]),
    ])
    assert (u0_grad_analytic == pytest.approx(u0_grad, rel=0.0, abs=1e-3))
    assert (u0_grad_np[:, 2:-2, 2:-2] == pytest.approx(u0_grad[:, 2:-2, 2:-2], rel=0.0, abs=1e-3))


@pytest.mark.parametrize("order", [2, 4, 6])
def test_torch_gradient_3d(order):
    lattice = Lattice(D3Q27, device='cpu', )
    flow = TaylorGreenVortex3D(resolution=100, reynolds_number=1, mach_number=0.05, lattice=lattice)
    grid = flow.grid
    p, u = flow.initial_solution(grid)
    dx = flow.units.convert_length_to_pu(1.0)
    u0_grad = torch_gradient(lattice.convert_to_tensor(u[0]), dx=dx, order=order).numpy()
    u0_grad_np = np.array(np.gradient(u[0], dx))
    u0_grad_analytic = np.array([
        np.cos(grid[0]) * np.cos(grid[1]) * np.cos(grid[2]),
        np.sin(grid[0]) * np.sin(grid[1]) * (-np.cos(grid[2])),
        np.sin(grid[0]) * (-np.cos(grid[1])) * np.sin(grid[2])
    ])
    assert np.allclose(u0_grad_analytic, u0_grad, rtol=0.0, atol=1e-3)
    assert np.allclose(u0_grad_np[:, 2:-2, 2:-2, 2:-2], u0_grad[:, 2:-2, 2:-2, 2:-2], rtol=0.0, atol=1e-3)


def test_grid_fine_to_coarse_2d():
    lattice = Lattice(D2Q9, 'cpu', dtype=torch.double)
    # streaming = StandardStreaming(lattice)

    flow_f = TaylorGreenVortex2D(40, 1600, 0.15, lattice)
    collision_f = BGKCollision(lattice, tau=flow_f.units.relaxation_parameter_lu)
    streaming_f = StandardStreaming(lattice)
    sim_f = Simulation(flow_f, lattice, collision_f, streaming_f)

    flow_c = TaylorGreenVortex2D(20, 1600, 0.15, lattice)
    collision_c = BGKCollision(lattice, tau=flow_c.units.relaxation_parameter_lu)
    streaming_c = StandardStreaming(lattice)
    sim_c = Simulation(flow_c, lattice, collision_c, streaming_c)

    f_c = grid_fine_to_coarse(lattice, sim_f.f, flow_f.units.relaxation_parameter_lu,
                              flow_c.units.relaxation_parameter_lu)

    p_init, u_init = flow_c.initial_solution(flow_c.grid)
    rho_init = lattice.convert_to_tensor(flow_c.units.convert_pressure_pu_to_density_lu(p_init))
    u_init = lattice.convert_to_tensor(flow_c.units.convert_velocity_to_lu(u_init))
    shear_c_init = lattice.shear_tensor(sim_c.f)
    shear_c = lattice.shear_tensor(f_c)

    assert torch.isclose(lattice.u(f_c), u_init).all()
    assert torch.isclose(lattice.rho(f_c), rho_init).all()
    assert torch.isclose(f_c, sim_c.f).all()
    assert torch.isclose(shear_c_init, shear_c).all()


def test_grid_fine_to_coarse_3d():
    lattice = Lattice(D3Q27, 'cpu', dtype=torch.double)

    flow_f = TaylorGreenVortex3D(40, 1600, 0.15, lattice)
    collision_f = BGKCollision(lattice, tau=flow_f.units.relaxation_parameter_lu)
    sim_f = Simulation(flow_f, lattice, collision_f)

    flow_c = TaylorGreenVortex3D(20, 1600, 0.15, lattice)
    collision_c = BGKCollision(lattice, tau=flow_c.units.relaxation_parameter_lu)
    sim_c = Simulation(flow_c, lattice, collision_c)

    f_c = grid_fine_to_coarse(
        lattice,
        sim_f.f,
        flow_f.units.relaxation_parameter_lu,
        flow_c.units.relaxation_parameter_lu
    )

    p_c_init, u_c_init = flow_c.initial_solution(flow_c.grid)
    rho_c_init = flow_c.units.convert_pressure_pu_to_density_lu(p_c_init)
    u_c_init = flow_c.units.convert_velocity_to_lu(u_c_init)
    shear_c_init = lattice.shear_tensor(sim_c.f)
    shear_c = lattice.shear_tensor(f_c)

    assert np.isclose(lattice.u(f_c).cpu().numpy(), u_c_init).all()
    assert np.isclose(lattice.rho(f_c).cpu().numpy(), rho_c_init).all()
    assert torch.isclose(f_c, sim_c.f).all()
    assert torch.isclose(shear_c_init, shear_c).all()


@pytest.mark.parametrize("Stencil", [D2Q9, D3Q27])
def test_pressure_poisson(dtype_device, Stencil):
    dtype, device = dtype_device
    lattice = Lattice(Stencil(), device, dtype)
    flow_class = TaylorGreenVortex2D if Stencil is D2Q9 else TaylorGreenVortex3D
    flow = flow_class(resolution=32, reynolds_number=100, mach_number=0.05, lattice=lattice)
    p0, u = flow.initial_solution(flow.grid)
    u = flow.units.convert_velocity_to_lu(lattice.convert_to_tensor(u))
    rho0 = flow.units.convert_pressure_pu_to_density_lu(lattice.convert_to_tensor(p0))
    rho = pressure_poisson(flow.units, u, torch.ones_like(rho0))
    pfinal = flow.units.convert_density_lu_to_pressure_pu(rho).cpu().numpy()
    print(p0.max(), p0.min(), p0.mean(), pfinal.max(), pfinal.min(), pfinal.mean())
    print((p0 - pfinal).max(), (p0 - pfinal).min())
    assert p0 == pytest.approx(pfinal, rel=0.0, abs=0.05)
