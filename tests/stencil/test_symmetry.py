from tests.common import *


def test_symmetry(fix_stencil):
    """Test if the stencil is symmetric"""
    assert np.array(fix_stencil.e).sum() == pytest.approx(0.0)
