"""
Test functions for collision models and related functions.
"""

from copy import copy
import torch
import pytest
import numpy as np
from lettuce import *
from tests.common import DummyFlow


@pytest.mark.parametrize("Transform", [D2Q9Lallemand, D2Q9Dellar])
def test_collision_fixpoint_2x_MRT(Transform, dtype_device):
    dtype, device = dtype_device
    context = Context(device=device, dtype=dtype)
    np.random.seed(1)  # arbitrary, but deterministic
    stencil = D2Q9()
    f = context.convert_to_tensor(np.random.random([stencil.q] + [3] *
                                                   stencil.d))
    f_old = copy(f)
    flow = DummyFlow(context, 1)
    collision = MRTCollision(Transform(stencil), np.array([0.5] * 9))
    f = collision(collision(flow))
    print(f.cpu().numpy(), f_old.cpu().numpy())
    assert f.cpu().numpy() == pytest.approx(f_old.cpu().numpy(), abs=1e-5)


def test_bgk_collision_devices(lattice2):
    if lattice2[0].stencil.D() != 2 and lattice2[0].stencil.D() != 3:
        pytest.skip("Test for 2D and 3D only!")

    def simulate(lattice):
        Flow = TaylorGreenVortex2D if lattice2[0].stencil.D() == 2 else (
            TaylorGreenVortex3D)
        flow = Flow(resolution=16, reynolds_number=10, mach_number=0.05,
                    lattice=lattice)

        collision = BGKCollision(lattice,
                                 tau=flow.units.relaxation_parameter_lu)
        streaming = NoStreaming(lattice)
        simulation = Simulation(flow=flow, lattice=lattice,
                                collision=collision, streaming=streaming)
        simulation.step(4)

        return simulation.f

    lattice0, lattice1 = lattice2
    f0 = simulate(lattice0).to(torch.device("cpu"))
    f1 = simulate(lattice1).to(torch.device("cpu"))
    error = torch.abs(f0 - f1).sum().data
    assert float(error) < 1.0e-8
