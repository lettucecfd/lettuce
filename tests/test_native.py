import pytest
import torch

from lettuce import StandardStreaming, Lattice, Obstacle2D, D2Q9, NoCollision, BGKCollision, NoStreaming, Simulation


def test_native_no_streaming_mask():
    """test if """

    if not torch.cuda.is_available():
        pytest.skip("Native test skipped")

    dtype = torch.float32
    device = "cuda:0"

    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = Obstacle2D(16, 16, 128, 0.1, lattice, 2)
    flow.mask[5, 5] = 1

    streaming = StandardStreaming(lattice)
    collision = NoCollision(lattice)

    simulation = Simulation(flow, lattice, collision, streaming)
    simulation.step(64)

    lattice_n = Lattice(D2Q9, dtype=dtype, device=device)
    flow_n = Obstacle2D(16, 16, 128, 0.1, lattice_n, 2)
    flow_n.mask[5, 5] = 1

    streaming_n = StandardStreaming(lattice_n)
    collision_n = NoCollision(lattice_n)

    simulation_n = Simulation(flow_n, lattice_n, collision_n, streaming_n)
    assert not (simulation_n.collide_and_stream == simulation_n.collide_and_stream_)
    simulation_n.step(64)

    error = torch.abs(simulation.f - simulation_n.f).sum().data
    assert error == 0.0


def test_native_no_collision_mask():
    """test if """

    if not torch.cuda.is_available():
        pytest.skip("Native test skipped")

    dtype = torch.float32
    device = "cuda:0"

    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = Obstacle2D(16, 16, 128, 0.1, lattice, 2)
    flow.mask[5, 5] = 1

    streaming = NoStreaming(lattice)
    collision = BGKCollision(lattice, 1.0)

    simulation = Simulation(flow, lattice, collision, streaming)
    simulation.step(64)

    lattice_n = Lattice(D2Q9, dtype=dtype, device=device)
    flow_n = Obstacle2D(16, 16, 128, 0.1, lattice_n, 2)
    flow_n.mask[5, 5] = 1

    streaming_n = NoStreaming(lattice_n)
    collision_n = BGKCollision(lattice_n, 1.0)

    simulation_n = Simulation(flow_n, lattice_n, collision_n, streaming_n)
    assert not (simulation_n.collide_and_stream == simulation_n.collide_and_stream_)
    simulation_n.step(64)

    error = torch.abs(simulation.f - simulation_n.f).sum().data
    assert error < 1.0e-24
