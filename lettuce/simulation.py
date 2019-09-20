"""Lattice Boltzmann Solver"""

from timeit import default_timer as timer
from lettuce import LettuceException, get_default_moment_transform, BGKInitialization, ExperimentalWarning, io
import pickle
import copy
import warnings
import torch


class Simulation:
    """High-level API for simulations."""
    def __init__(self, flow, lattice, collision, streaming, boundary):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.boundary = boundary
        self.i = 0

        grid = flow.grid
        p, u = flow.initial_solution(grid)
        assert list(p.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        self.reporters = []

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        for i in range(num_steps):
            self.i += 1
            self.f = self.streaming(self.f)
            self.f = self.collision(self.f)
            self.f = self.boundary(self.f)
            for reporter in self.reporters:
                reporter(self.i, self.i, self.f)
            if i % 10 == 0:
                io.write_vtk("output_vtk", [self.flow.resolution*2, self.flow.resolution, 1], self.lattice.convert_to_numpy(self.collision.lattice.u(self.f)), self.lattice.convert_to_numpy(self.collision.lattice.rho(self.f)), str(i))
                print(i)
        end = timer()
        seconds = end-start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def initialize(self, max_num_steps=500, tol_pressure=0.001):
        """Iterative initialization to get moments consistent with the initial velocity.

        Using the initialization does not better TGV convergence. Maybe use a better scheme?
        """
        warnings.warn("Iterative initialization does not work well and solutions may diverge. Use with care. "
                      "At some point, we may need to implement a better scheme.",
                      ExperimentalWarning)
        transform = get_default_moment_transform(self.lattice)
        collision = BGKInitialization(self.lattice, self.flow, transform)
        streaming = self.streaming
        p_old = 0
        for i in range(max_num_steps):
            self.f = streaming(self.f)
            self.f = collision(self.f)
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(self.f))
            if (torch.max(torch.abs(p-p_old))) < tol_pressure:
                break
            p_old = copy.deepcopy(p)
        return i

    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)

