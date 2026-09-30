import os
import h5py
from tests.conftest import *


def test_HDF5Reporter(tmpdir):
    step = 3
    context = Context(device='cpu')
    flow = TaylorGreenVortex(context=context,
                             resolution=[16, 16],
                             reynolds_number=10,
                             mach_number=0.05)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = Simulation(flow=flow,
                            collision=collision,
                            reporter=[])
    hdf5_reporter = HDF5Reporter(flow=flow,
                                 collision=collision,
                                 interval=step,
                                 filebase=tmpdir / "output")
    simulation.reporter.append(hdf5_reporter)
    simulation(step)
    assert os.path.isfile(tmpdir / "output.h5")

    dataset_train = LettuceDataset(filebase=tmpdir / "output.h5",
                                   target=True)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=False)
    print(dataset_train)
    for (f, target, idx) in train_loader:
        assert idx in (0, 1, 2)
        assert f.shape == (1, 9, 16, 16)
        assert target.shape == (1, 9, 16, 16)


def test_HDF5Reporter_keeps_precision(tmpdir, fix_dtype):
    """The stored populations must not be truncated to a lower precision than
    the simulation runs in (h5py defaults to float32 without an explicit
    dtype)."""
    context = Context(device='cpu', dtype=fix_dtype)
    flow = TaylorGreenVortex(context=context,
                             resolution=[16, 16],
                             reynolds_number=10,
                             mach_number=0.05)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = Simulation(flow=flow, collision=collision, reporter=[])
    simulation.reporter.append(HDF5Reporter(flow=flow,
                                            collision=collision,
                                            interval=1,
                                            filebase=tmpdir / "output"))
    f_before = context.convert_to_ndarray(flow.f)
    simulation(1)

    with h5py.File(tmpdir / "output.h5", "r") as fs:
        assert fs["f"].dtype == f_before.dtype
        np.testing.assert_array_equal(fs["f"][0], f_before)
