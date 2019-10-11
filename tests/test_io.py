
import pytest
import os
from lettuce import TaylorGreenVortex2D, Lattice, D2Q9, write_image, BGKCollision, StandardStreaming,Simulation,EnergyReporter
import numpy as np


def test_write_image(tmpdir):
    pytest.skip("matplotlib not working")
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    p, u = flow.initial_solution(flow.grid)
    write_image(tmpdir/"p.png", p[0])
    print(tmpdir/"p.png")
    assert os.path.isfile(tmpdir/"p.png")

def test_energy_reporter(dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = TaylorGreenVortex2D(16, 10000, 0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    kinE_reporter = EnergyReporter(lattice, flow, interval=1, out=None)
    simulation.reporters.append(kinE_reporter)
    simulation.step(2)
    assert(np.asarray(kinE_reporter.out)[1,1] == pytest.approx(np.asarray(kinE_reporter.out)[0,1],abs=0.1))


