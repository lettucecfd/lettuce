
import numpy as np
from lettuce import Lattice, D2Q9, D3Q27, TaylorGreenVortex2D, TaylorGreenVortex3D, torch_gradient
import pytest


@pytest.mark.parametrize("order", [2,4,6])
def test_torch_gradient_2d(order):
    lattice = Lattice(D2Q9, device='cpu',)
    flow = TaylorGreenVortex2D(resolution=100, reynolds_number=1, mach_number=0.05, lattice=lattice)
    grid = flow.grid
    p, u = flow.initial_solution(grid)
    dx = grid[0][0][1]-grid[0][0][0]
    u0_grad = torch_gradient(lattice.convert_to_tensor(u[0]), dx=dx, order=order).numpy()
    u0_grad_np = np.gradient(u[0],dx)
    u0_grad_analytic = np.array([
        -np.sin(grid[0])*np.sin(grid[1]),
        np.cos(grid[0])*np.cos(grid[1]),
    ])
    assert (u0_grad_analytic[1,1,:] == pytest.approx(u0_grad[0,1,:], rel=2))
    assert (u0_grad_np[0][1,:] == pytest.approx(u0_grad[0, 1, :], rel=2))


@pytest.mark.parametrize("order", [2,4,6])
def test_torch_gradient_3d(order):
    lattice = Lattice(D3Q27, device='cpu', )
    flow = TaylorGreenVortex3D(resolution=100, reynolds_number=1, mach_number=0.05, lattice=lattice)
    grid = flow.grid
    p, u = flow.initial_solution(grid)
    dx = grid[0][0][1][0] - grid[0][0][0][0]
    u0_grad = torch_gradient(lattice.convert_to_tensor(u[0]), dx=dx, order=order).numpy()
    u0_grad_np = np.gradient(u[0],dx)
    u0_grad_analytic = np.array([
        np.cos(grid[0]) * np.cos(grid[1]) * np.cos(grid[2]),
        np.sin(grid[0]) * np.sin(grid[1]) * (-np.cos(grid[2])),
        np.sin(grid[0]) * (-np.cos(grid[1])) * np.sin(grid[2])
    ])
    assert (u0_grad_analytic[1,2,:,2] == pytest.approx(u0_grad[0,2,:,2], rel=2))
    assert (u0_grad_np[0][20,:,0] == pytest.approx(u0_grad[0, 20, :, 0], rel=2))
