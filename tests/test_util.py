
import numpy as np
import torch
from lettuce import Lattice, D2Q9, D3Q27, TaylorGreenVortex2D, TaylorGreenVortex3D, torch_gradient,grid_fine_to_course
from lettuce import BGKCollision, Simulation
import pytest


@pytest.mark.parametrize("order", [2,4,6])
def test_torch_gradient_2d(order):
    lattice = Lattice(D2Q9, device='cpu',)
    flow = TaylorGreenVortex2D(resolution=100, reynolds_number=1, mach_number=0.05, lattice=lattice)
    grid = flow.grid
    p, u = flow.initial_solution(grid)
    dx = grid[0][0][1]-grid[0][0][0]
    u0_grad = torch_gradient(lattice.convert_to_tensor(u[0]), dx=dx, order=order).numpy()
    u0_grad_analytic = np.array([
        -np.sin(grid[0])*np.sin(grid[1]),
        np.cos(grid[0])*np.cos(grid[1]),
    ])
    assert (u0_grad_analytic[0,1,:] == pytest.approx(u0_grad[0,1,:], rel=2))


@pytest.mark.parametrize("order", [2,4,6])
def test_torch_gradient_3d(order):
    lattice = Lattice(D3Q27, device='cpu', )
    flow = TaylorGreenVortex3D(resolution=100, reynolds_number=1, mach_number=0.05, lattice=lattice)
    grid = flow.grid
    p, u = flow.initial_solution(grid)
    dx = grid[0][0][1] - grid[0][0][0]
    u0_grad = torch_gradient(lattice.convert_to_tensor(u[0]), dx=dx, order=order).numpy()
    u0_grad_analytic = np.array([
        np.cos(grid[0]) * np.cos(grid[1]) * np.cos(grid[2]),
        np.sin(grid[0]) * np.sin(grid[1]) * (-np.cos(grid[2])),
        np.sin(grid[0]) * (-np.cos(grid[1])) * np.sin(grid[2])
    ])
    assert (u0_grad_analytic[0,0,:,0] == pytest.approx(u0_grad[0,0,:,0], rel=2))

def test_grid_fine_to_couse_2d():
    lattice = Lattice(D2Q9,'cpu',dtype=torch.double)
    # streaming = StandardStreaming(lattice)

    flow_f = TaylorGreenVortex2D(120,1600,0.15,lattice)
    collision_f = BGKCollision(lattice,tau=flow_f.units.relaxation_parameter_lu)
    sim_f = Simulation(flow_f,lattice,collision_f,streaming=None)

    flow_c = TaylorGreenVortex2D(60,1600,0.15,lattice)
    collision_c = BGKCollision(lattice,tau=flow_c.units.relaxation_parameter_lu)
    sim_c = Simulation(flow_c,lattice,collision_c,streaming=None)

    f_c = grid_fine_to_course(lattice,sim_f.f,flow_f.units.relaxation_parameter_lu,flow_c.units.relaxation_parameter_lu)

    p_init, u_init = flow_c.initial_solution(flow_c.grid)
    rho_init = lattice.convert_to_tensor(flow_c.units.convert_pressure_pu_to_density_lu(p_init))
    u_init = lattice.convert_to_tensor(flow_c.units.convert_velocity_to_lu(u_init))

    assert (lattice.u(f_c).numpy() == pytest.approx(u_init.numpy()))
    assert (lattice.rho(f_c).numpy() == pytest.approx(rho_init.numpy()))
    assert (f_c.numpy() == pytest.approx(sim_c.f.numpy()))

def test_grid_fine_to_couse_3d():
    lattice = Lattice(D3Q27,'cpu',dtype=torch.double)

    flow_f = TaylorGreenVortex3D(120,1600,0.15,lattice)
    collision_f = BGKCollision(lattice,tau=flow_f.units.relaxation_parameter_lu)
    sim_f = Simulation(flow_f,lattice,collision_f,streaming=None)

    flow_c = TaylorGreenVortex3D(60,1600,0.15,lattice)
    collision_c = BGKCollision(lattice,tau=flow_c.units.relaxation_parameter_lu)
    sim_c = Simulation(flow_c,lattice,collision_c,streaming=None)

    f_c = grid_fine_to_course(lattice,sim_f.f,flow_f.units.relaxation_parameter_lu,flow_c.units.relaxation_parameter_lu)

    p_c_init, u_c_init = flow_c.initial_solution(flow_c.grid)
    rho_c_init = flow_c.units.convert_pressure_pu_to_density_lu(p_c_init)
    u_c_init = flow_c.units.convert_velocity_to_lu(u_c_init)

    assert lattice.u(f_c).numpy() == pytest.approx(u_c_init)
    assert lattice.rho(f_c).numpy() == pytest.approx(rho_c_init)
    assert f_c.numpy() == pytest.approx(sim_c.f.numpy())
