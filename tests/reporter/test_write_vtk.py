from tests.common import *
from lettuce.ext._reporter.vtk_reporter import write_vtk
import os

def test_write_vtk(tmpdir):
    context = Context(device='cpu')
    flow = TaylorGreenVortex(context, resolution=[16, 16], reynolds_number=10,
                             mach_number=0.05)
    p, u = flow.initial_pu()
    point_dict = {"p": context.convert_to_ndarray(p[0, ..., None])}
    write_vtk(point_dict, id=1, filename_base=tmpdir / "output")
    assert os.path.isfile(tmpdir / "output_00000001.vtr")