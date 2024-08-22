from tests.conftest import *


def test_first_zero(fix_stencil):
    """Test that the zeroth velocity is 0."""
    assert (fix_stencil.e[0]
            == pytest.approx(np.zeros_like(fix_stencil.e[0])))
