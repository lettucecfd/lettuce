"""Lattice Boltzmann Solver"""

import pickle
import warnings
from copy import deepcopy
from timeit import default_timer as timer

import numpy as np
import torch

from . import *
from .native_generator import NativeStencil, Generator


class Simulation:
    """High-level API for simulations.

    Attributes
    ----------
    reporters : list
        A list of reporters. Their call functions are invoked after every simulation step (and before the first one).

    """

    def __init__(self, flow, lattice, collision, streaming=None, use_native=True):
        self.flow = flow
        self.lattice: Lattice = lattice
        self.collision: Collision = collision
        self.streaming: Streaming = StandardStreaming(lattice) if streaming is None else streaming
        self.i = 0

        grid = flow.grid
        p, u = flow.initial_solution(grid)
        assert list(p.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.d] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.d] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        self.reporters = []

        # Define masks, where the collision or streaming are not applied
        x = flow.grid
        no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0], dtype=bool)).to(dtype=torch.bool)
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool)).to(dtype=torch.bool)

        # Apply boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                no_collision_mask = no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
            if hasattr(boundary, "make_no_stream_mask"):
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)

        if no_stream_mask.any():
            self.streaming.no_streaming_mask = no_stream_mask.to(torch.uint8)
        if no_collision_mask.any():
            self.collision.no_collision_mask = no_collision_mask.to(torch.uint8)

        self.stream_and_collide = Simulation.stream_and_collide_

        if use_native:

            if str(lattice.device) == 'cpu':
                return

            if not self.streaming.native_available():
                print('streaming is not native available')
                return
            if not self.collision.native_available():
                print('collision is not native available')
                return

            native_stencil = NativeStencil(self.lattice.stencil)
            native_streaming = self.streaming.create_native()
            native_collision = self.collision.create_native()

            generator = Generator(native_stencil, native_streaming, native_collision)

            stream_and_collide = generator.resolve()
            if stream_and_collide is None:
                _val = generator.generate()
                _dir = generator.format(_val)
                generator.install(_dir)

                stream_and_collide = generator.resolve()
                if stream_and_collide is None:
                    print('failed to install native extension!')
                    return

            self.stream_and_collide = stream_and_collide
            if 'noStreaming' not in native_streaming.name:  # TODO find a better way of storing f_next
                self.f_next = torch.empty(self.f.shape, dtype=self.lattice.dtype, device=self.f.get_device())

    @staticmethod
    def stream_and_collide_(self):
        self.f = self.streaming(self.f)
        # Perform the collision routine everywhere, expect where the no_collision_mask is true
        if self.collision.no_collision_mask is not None:
            self.f = torch.where(self.collision.no_collision_mask, self.f, self.collision(self.f))
        else:
            self.collision(self.f)

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            self.i += 1
            self.stream_and_collide(self)
            for boundary in self._boundaries:
                self.f = boundary(self.f)
            self._report()

        end = timer()
        seconds = end - start

        num_grid_points = self.f.numel() / self.lattice.stencil.q()

        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def _report(self):
        for reporter in self.reporters:
            reporter(self.i, self.flow.units.convert_time_to_pu(self.i), self.f)

    def initialize(self, max_num_steps=500, tol_pressure=0.001):
        """Iterative initialization to get moments consistent with the initial velocity.

        Using the initialization does not better TGV convergence. Maybe use a better scheme?
        """
        warnings.warn("Iterative initialization does not work well and solutions may diverge. Use with care. "
                      "Use initialize_f_neq instead.",
                      ExperimentalWarning)
        transform = get_default_moment_transform(self.lattice)
        collision = BGKInitialization(self.lattice, self.flow, transform)
        streaming = self.streaming
        p_old = 0
        for i in range(max_num_steps):
            self.f = streaming(self.f)
            self.f = collision(self.f)
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(self.f))
            if (torch.max(torch.abs(p - p_old))) < tol_pressure:
                break
            p_old = deepcopy(p)
        return i

    def initialize_pressure(self, max_num_steps=100000, tol_pressure=1e-6):
        """Reinitialize equilibrium distributions with pressure obtained by a Jacobi solver.
        Note that this method has to be called before initialize_f_neq.
        """
        u = self.lattice.u(self.f)
        rho = pressure_poisson(
            self.flow.units,
            self.lattice.u(self.f),
            self.lattice.rho(self.f),
            tol_abs=tol_pressure,
            max_num_steps=max_num_steps
        )
        self.f = self.lattice.equilibrium(rho, u)

    def initialize_f_neq(self):
        """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
        See Krüger et al. (2017).
        """
        rho = self.lattice.rho(self.f)
        u = self.lattice.u(self.f)

        grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
        grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
        S = torch.cat([grad_u0, grad_u1])

        if self.lattice.d == 3:
            grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
            S = torch.cat([S, grad_u2])

        Pi_1 = 1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2
        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.d, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)
        Pi_1_Q = self.lattice.einsum('ab,iab->i', [Pi_1, Q])
        fneq = self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q])

        feq = self.lattice.equilibrium(rho, u)
        self.f = feq + fneq

    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)
