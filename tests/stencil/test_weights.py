from tests.conftest import *


def test_weights(fix_stencil):
    """Test if the sum of all weights equals one."""
    assert np.array(fix_stencil.w).sum() == pytest.approx(1.0)
