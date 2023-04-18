"""
Tests for streaming operators.
"""

from lettuce import *
from lettuce.flows import *

import torch
import pytest
import copy


def test_standard_streaming_x3(f_all_lattices):
    """Streaming three times on a 3^D grid gives the original distribution functions."""
    f, lattice = f_all_lattices
    f_old = copy.copy(f.cpu().numpy())
    streaming = StandardStreaming(lattice)
    f = streaming(streaming(streaming(f)))
    assert f.cpu().numpy() == pytest.approx(f_old)


def test_standard_streaming_hardcoded_2D(lattice):
    if not (lattice.stencil.D() == 2):
        pytest.skip("Custom Test for 2D stencils")

    dummy_flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = NoCollision(lattice)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(dummy_flow, lattice, collision, streaming)

    if hasattr(simulation, 'f_next'):
        simulation.f_next = torch.zeros((lattice.stencil.Q(), 16, 16), dtype=lattice.dtype, device=lattice.device)

    assert hasattr(simulation, 'f')
    f = torch.zeros((lattice.stencil.Q(), 16, 16), dtype=lattice.dtype, device=lattice.device)
    for q in range(lattice.stencil.Q()):
        f[q, 1, 1] = q + 1
    simulation.f = f

    simulation.collide_and_stream(simulation)
    f = simulation.f

    for q in range(lattice.stencil.Q()):
        assert f[q, 1 + lattice.stencil.e[q, 0], 1 + lattice.stencil.e[q, 1]] == q + 1


def test_standard_streaming_hardcoded_3D(lattice):
    if not (lattice.stencil.D() == 3):
        pytest.skip("Custom Test for 3D stencils")

    dummy_flow = TaylorGreenVortex3D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = NoCollision(lattice)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(dummy_flow, lattice, collision, streaming)

    if hasattr(simulation, 'f_next'):
        simulation.f_next = torch.zeros((lattice.stencil.Q(), 16, 16, 16), dtype=lattice.dtype, device=lattice.device)

    assert hasattr(simulation, 'f')
    f = torch.zeros((lattice.stencil.Q(), 16, 16, 16), dtype=lattice.dtype, device=lattice.device)
    for q in range(lattice.stencil.Q()):
        f[q, 1, 1, 1] = q + 1
    simulation.f = f

    simulation.collide_and_stream(simulation)
    f = simulation.f

    for q in range(lattice.stencil.Q()):
        assert f[q, 1 + lattice.stencil.e[q, 0], 1 + lattice.stencil.e[q, 1], 1 + lattice.stencil.e[q, 2]] == q + 1


def test_standard_streaming_devices(lattice2):
    if lattice2[0].stencil.D() != 2 and lattice2[0].stencil.D() != 3:
        pytest.skip("Test for 2D and 3D only!")

    def simulate(lattice):
        Flow = TaylorGreenVortex2D if lattice2[0].stencil.D() == 2 else TaylorGreenVortex3D
        flow = Flow(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)

        collision = NoCollision(lattice)
        streaming = StandardStreaming(lattice)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
        simulation.step(4)

        return simulation.f

    lattice0, lattice1 = lattice2
    f0 = simulate(lattice0).to(torch.device("cpu"))
    f1 = simulate(lattice1).to(torch.device("cpu"))

    error = torch.abs(f0 - f1).sum().data
    assert float(error) == 0.0
