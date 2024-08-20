from tests.common import *
import os

def test_write_image(tmpdir):
    # pytest.skip("matplotlib not working")
    context = Context()
    flow = TaylorGreenVortex(context=context,
                             resolution=[16, 16],
                             reynolds_number=10,
                             mach_number=0.05)
    p, _ = flow.initial_pu()
    write_image(tmpdir / "p.png", context.convert_to_ndarray(p[0]))
    print(tmpdir / "p.png")
    assert os.path.isfile(tmpdir / "p.png")