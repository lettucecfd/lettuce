import pytest
import numpy as np


def test_weights(lattice):
    """Test if the sum of all weights equals one."""
    assert sum(lattice.w.cpu().numpy()) == pytest.approx(1.0)


def test_first_zero(lattice):
    """Test that the zeroth velocity is 0."""
    assert lattice.stencil.e[0] == pytest.approx(np.zeros_like(lattice.stencil.e[0]))
