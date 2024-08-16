"""
Test boundary conditions.
"""

from lettuce import *
from lettuce.ext import *

import pytest

from copy import copy
import numpy as np
import torch


def test_masks(dtype_device):
    """test if masks are applied from boundary conditions"""
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = Obstacle((16, 16), 100, 0.1, lattice, 2)
    flow.mask[1, 1] = 1
    streaming = StandardStreaming(lattice)
    collision = BGKCollision(lattice, 1.0)
    simulation = Simulation(flow, lattice, collision, streaming)
    assert simulation.streaming.no_stream_mask.any()
    assert simulation.collision.no_collision_mask.any()


def test_equilibrium_pressure_outlet(dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device)

    class MyObstacle(Obstacle):
        @property
        def boundaries(self, *args):
            x, y = self.grid
            return [
                EquilibriumBoundaryPU(
                    np.abs(x) < 1e-6, self.units.lattice, self.units,
                    np.array([self.units.characteristic_velocity_pu, 0])
                ),
                EquilibriumOutletP(self.units.lattice, [0, -1]),
                EquilibriumOutletP(self.units.lattice, [0, 1]),
                EquilibriumOutletP(self.units.lattice, [1, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)
            ]

    flow = MyObstacle((32, 32), reynolds_number=10, mach_number=0.1,
                      lattice=lattice, domain_length_x=3)
    mask = np.zeros_like(flow.grid[0], dtype=bool)
    mask[10:20, 10:20] = 1
    flow.mask = mask
    simulation = Simulation(flow, lattice,
                            RegularizedCollision(
                                lattice,
                                flow.units.relaxation_parameter_lu),
                            StandardStreaming(lattice))
    simulation.step(30)
    rho = lattice.rho(simulation.f)
    u = lattice.u(simulation.f)
    feq = lattice.equilibrium(torch.ones_like(rho), u)
    p = flow.units.convert_density_lu_to_pressure_pu(rho)
    zeros = torch.zeros_like(p[0, -1, :])
    assert torch.allclose(zeros, p[:, -1, :], rtol=0, atol=1e-4)
    assert torch.allclose(zeros, p[:, :, 0], rtol=0, atol=1e-4)
    assert torch.allclose(zeros, p[:, :, -1], rtol=0, atol=1e-4)
    assert torch.allclose(feq[:, -1, 1:-1], feq[:, -2, 1:-1])
