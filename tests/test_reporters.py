
import pytest
import os
from lettuce import TaylorGreenVortex2D, TaylorGreenVortex3D, Lattice, D3Q27, D2Q9, write_image, BGKCollision, StandardStreaming, Simulation, DecayingTurbulence
from lettuce.reporters import write_vtk, VTKReporter,EnstrophyReporter,EnergyReporter,MaxUReporter,SpectrumReporter
import numpy as np
import torch


def test_write_image(tmpdir):
    pytest.skip("matplotlib not working")
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    p, u = flow.initial_solution(flow.grid)
    write_image(tmpdir/"p.png", p[0])
    print(tmpdir/"p.png")
    assert os.path.isfile(tmpdir/"p.png")


@pytest.mark.parametrize("Reporter", [EnstrophyReporter, EnergyReporter, MaxUReporter])
@pytest.mark.parametrize("Case", [TaylorGreenVortex2D,TaylorGreenVortex3D])
def test_generic_reporters(Reporter, Case, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = Case(32, 10000, 0.05, lattice=lattice)
    if Case == TaylorGreenVortex3D:
        lattice = Lattice(D3Q27, dtype=dtype, device=device)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    reporter = Reporter(lattice, flow, interval=1, out=None)
    simulation.reporters.append(reporter)
    simulation.step(2)
    values = np.asarray(reporter.out)
    assert(values[1,1] == pytest.approx(values[0,1], rel=0.05))


def test_write_vtk(tmpdir):
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    p, u = flow.initial_solution(flow.grid)
    point_dict = {}
    point_dict["p"] = p[0, ..., None]
    write_vtk(point_dict, id=1, filename_base=tmpdir/"output")
    assert os.path.isfile(tmpdir/"output_00000001.vtr")


def test_vtk_reporter(tmpdir):
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    vtk_reporter = VTKReporter(lattice, flow, interval=1, filename_base=tmpdir/"output")
    simulation.reporters.append(vtk_reporter)
    simulation.step(2)
    assert os.path.isfile(tmpdir/"output_00000000.vtr")
    assert os.path.isfile(tmpdir/"output_00000001.vtr")


@pytest.mark.parametrize("Flow", [TaylorGreenVortex2D, TaylorGreenVortex3D, 'DecayingTurbulence2D', 'DecayingTurbulence3D'])
def test_EnergySpectrumReporter(tmpdir, Flow):
    lattice = Lattice(D2Q9, device='cpu')
    if Flow == TaylorGreenVortex3D or Flow is 'DecayingTurbulence3D':
        lattice = Lattice(D3Q27, device='cpu')
    if Flow is 'DecayingTurbulence2D' or Flow is 'DecayingTurbulence3D':
        Flow = DecayingTurbulence
    flow = Flow(resolution=20, reynolds_number=1600, mach_number=0.01, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.reporters.append(SpectrumReporter(lattice, flow, out=None))
    simulation.reporters.append(EnergyReporter(lattice, flow, out=None))
    simulation.step(1)
    spectrum = simulation.reporters[0].out
    energy = simulation.reporters[1].out

    if Flow == DecayingTurbulence:
        ek_ref, _ = flow.energy_spectrum
        assert (spectrum[0][1] == pytest.approx(ek_ref, rel= 0.0, abs=0.1))
    if Flow == TaylorGreenVortex2D or Flow == TaylorGreenVortex3D:
        ek_max = sorted(spectrum[0][1], reverse=True)
        assert ek_max[0]*1e-5>ek_max[1]
    assert (energy[0][1] == pytest.approx(torch.sum(spectrum[0][1]).numpy(), rel= 0.1, abs=0.0))



