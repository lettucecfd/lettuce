import os

from tests.conftest import *


def test_vtk_reporter_mask(tmpdir):
    flow = PoiseuilleFlow2D(context=Context(), resolution=16,
                            reynolds_number=10, mach_number=0.05)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = Simulation(flow, collision, [])
    vtk_reporter = VTKReporter(interval=1,
                               filename_base=tmpdir / "output")
    simulation.reporter.append(vtk_reporter)
    vtk_reporter.output_mask(simulation)
    simulation(2)
    assert os.path.isfile(tmpdir / "output_mask.vtr")
    assert os.path.isfile(tmpdir / "output_00000000.vtr")
    assert os.path.isfile(tmpdir / "output_00000001.vtr")
