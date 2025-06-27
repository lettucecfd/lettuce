import os
from tests.conftest import *


def test_high_ma_reporter(tmpdir):
    flow = Obstacle(context=Context(), resolution=[16, 16],
                    reynolds_number=10, mach_number=0.2, domain_length_x=16)
    flow.mask = ((2 < flow.grid[0]) & (flow.grid[0] < 10)
                 & (2 < flow.grid[1]) & (flow.grid[1] < 10))
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    reporter = HighMaReporter(1, outdir=tmpdir, vtk=True)
    simulation = BreakableSimulation(flow, collision, [reporter])
    simulation(100)
    assert os.path.isfile(tmpdir / "HighMa_reporter.txt")
    assert os.path.isfile(tmpdir / "HighMa_fail_00000013.vtr")
    assert flow.i > 100
    assert reporter.failed_iteration == 13
