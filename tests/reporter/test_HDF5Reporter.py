import os
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
