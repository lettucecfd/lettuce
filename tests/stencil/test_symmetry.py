from tests.common import *


def test_symmetry(fix_stencil):
    """Test if the stencil is symmetric"""
    assert sum(np.array(fix_stencil.e)) == pytest.approx(0.0)
