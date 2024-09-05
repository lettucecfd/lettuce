import os
from tests.conftest import *


def test_nan_reporter(tmpdir):
    flow = TaylorGreenVortex(context=Context(), resolution=[16, 16],
                            reynolds_number=1e12, mach_number=1)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    reporter = NaNReporter(10, outdir=tmpdir, vtk=True)
    simulation = BreakableSimulation(flow, collision, [reporter])
    simulation(100)
    assert os.path.isfile(tmpdir / "NaN_reporter.txt")
    print(os.listdir(tmpdir))
    assert os.path.isfile(tmpdir / "NaN_fail_00000070.vtr") or os.path.isfile(tmpdir / "NaN_fail_00000080.vtr")
    assert flow.i > 100
    assert reporter.failed_iteration in [70, 80]
