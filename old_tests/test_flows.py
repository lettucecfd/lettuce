import pytest
import numpy as np
import torch
from lettuce import TaylorGreenVortex2D, TaylorGreenVortex3D, CouetteFlow2D, D2Q9, D3Q27, DoublyPeriodicShear2D
from lettuce import torch_gradient, DecayingTurbulence
from lettuce import Lattice, Simulation, BGKCollision, BGKInitialization, StandardStreaming
from lettuce import Obstacle2D, Obstacle3D
from lettuce.flows.poiseuille import PoiseuilleFlow2D
from lettuce import Context



@pytest.mark.parametrize("_stencil", [D2Q9, D3Q27])
def test_obstacle(stencil, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(stencil, dtype=dtype, device=device)

    nx = 32
    ny = 16
    nz = 16

    if stencil is D2Q9:
        mask = np.zeros([nx, ny])
        mask[3:6, 3:6] = 1
        flow = Obstacle2D(nx, ny, 100, 0.1, lattice=lattice, char_length_lu=3)
    if stencil is D3Q27:
        mask = np.zeros([nx, ny, nz])
        mask[3:6, 3:6, :] = 1
        flow = Obstacle3D(nx, ny, nz, 100, 0.1, lattice=lattice, char_length_lu=3)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    flow.mask = mask != 0
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    simulation.step(2)
