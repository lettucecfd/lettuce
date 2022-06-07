import pytest
import os
from lettuce import TaylorGreenVortex2D, TaylorGreenVortex3D, PoiseuilleFlow2D, Lattice, D3Q27, D2Q9, write_image, \
    BGKCollision, StandardStreaming, Simulation, DecayingTurbulence
from lettuce.reporters import write_vtk, VTKReporter, ObservableReporter
from lettuce.datautils import HDF5Reporter, LettuceDataset
from lettuce.observables import Enstrophy, EnergySpectrum, MaximumVelocity, IncompressibleKineticEnergy, Mass
import numpy as np
import torch


def test_write_image(tmpdir):
    pytest.skip("matplotlib not working")
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    p, u = flow.initial_solution(flow.grid)
    write_image(tmpdir / "p.png", p[0])
    print(tmpdir / "p.png")
    assert os.path.isfile(tmpdir / "p.png")


@pytest.mark.parametrize("Observable", [Enstrophy, EnergySpectrum, MaximumVelocity, IncompressibleKineticEnergy, Mass])
@pytest.mark.parametrize("Case", [TaylorGreenVortex2D, TaylorGreenVortex3D])
def test_generic_reporters(Observable, Case, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device, use_native=False)  # TODO use_native Fails here
    flow = Case(32, 10000, 0.05, lattice=lattice)
    if Case == TaylorGreenVortex3D:
        lattice = Lattice(D3Q27, dtype=dtype, device=device, use_native=False)  # TODO use_native Fails here
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    reporter = ObservableReporter(Observable(lattice, flow), interval=1, out=None)
    simulation.reporters.append(reporter)
    simulation.step(2)
    values = np.asarray(reporter.out)
    if Observable is EnergySpectrum:
        assert values[1, 2:] == pytest.approx(values[0, 2:], rel=0.0, abs=values[0, 2:].sum() / 10)
    else:
        assert values[1, 2] == pytest.approx(values[0, 2], rel=0.05)


def test_write_vtk(tmpdir):
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    p, u = flow.initial_solution(flow.grid)
    point_dict = {"p": p[0, ..., None]}
    write_vtk(point_dict, id=1, filename_base=tmpdir / "output")
    assert os.path.isfile(tmpdir / "output_00000001.vtr")


def test_vtk_reporter_no_mask(tmpdir):
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    vtk_reporter = VTKReporter(lattice, flow, interval=1, filename_base=tmpdir / "output")
    simulation.reporters.append(vtk_reporter)
    simulation.step(2)
    assert os.path.isfile(tmpdir / "output_00000000.vtr")
    assert os.path.isfile(tmpdir / "output_00000001.vtr")


def test_vtk_reporter_mask(tmpdir):
    lattice = Lattice(D2Q9, "cpu")
    flow = PoiseuilleFlow2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    vtk_reporter = VTKReporter(lattice, flow, interval=1, filename_base=tmpdir / "output")
    simulation.reporters.append(vtk_reporter)
    vtk_reporter.output_mask(simulation.no_collision_mask)
    simulation.step(2)
    assert os.path.isfile(tmpdir / "output_mask.vtr")
    assert os.path.isfile(tmpdir / "output_00000000.vtr")
    assert os.path.isfile(tmpdir / "output_00000001.vtr")


def test_HDF5Reporter(tmpdir):
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice=lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice=lattice)
    simulation = Simulation(flow=flow,
                            lattice=lattice,
                            collision=collision,
                            streaming=streaming)
    hdf5_reporter = HDF5Reporter(
        flow=flow,
        collision=collision,
        interval=1,
        filebase= tmpdir / "output")
    simulation.reporters.append(hdf5_reporter)
    simulation.step(3)
    assert os.path.isfile(tmpdir / "output.h5")

    dataset_train = LettuceDataset(
        filebase=tmpdir / "output.h5",
        target=True)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=False)
    print(dataset_train)
    for (f, target, idx) in train_loader:
        assert idx in (0,1,2)
        assert f.shape == (1,9,16,16)
        assert target.shape == (1,9,16,16)

