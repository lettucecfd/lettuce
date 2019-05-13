"""Lattice Boltzmann Solver"""

from timeit import default_timer as timer
import torch


class Simulation(object):
    """High-level API for simulations."""
    def __init__(self, flow, lattice, collision, streaming):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming

        p, u = flow.initial_solution(flow.grid)
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.quadratic_equilibrium(rho, lattice.convert_to_tensor(u))

        self.reporters = []

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        for i in range(num_steps):
            self.f = self.streaming(self.f)
            self.f = self.collision(self.f)
            for reporter in self.reporters:
                reporter(i, i, self.f)
        end = timer()
        seconds = end-start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def write_checkpoint(self, filename):
        """TODO:Write f as np.array using pickle module."""
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """TODO:Load f as np.array using pickle module."""
        raise NotImplementedError
