
import pytest
import os
from lettuce import TaylorGreenVortex2D, Lattice, D2Q9, write_image
from lettuce.io import write_vtk, VTKReporter


def test_write_image(tmpdir):
    pytest.skip("matplotlib not working")
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    p, u = flow.initial_solution(flow.grid)
    write_image(tmpdir/"p.png", p[0])
    print(tmpdir/"p.png")
    assert os.path.isfile(tmpdir/"p.png")

def test_write_vtk(tmpdir):
    lattice = Lattice(D2Q9, "cpu")
    flow = TaylorGreenVortex2D(resolution=16, reynolds_number=10, mach_number=0.05, lattice=lattice)
    p, u = flow.initial_solution(flow.grid)
    point_dict = {}
    point_dict["p"] = p[0, ..., None]
    write_vtk(point_dict, id=1, filename_base=tmpdir/"output")
    assert os.path.isfile(tmpdir/"output_00000001.vtr")