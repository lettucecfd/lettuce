
import pytest
import numpy as np


def test_opposite(lattice):
    """Test if the opposite field holds the index of the opposite direction."""
    assert lattice.e[lattice.stencil.opposite].cpu().numpy() == pytest.approx(-lattice.e.cpu().numpy())


def test_symmetry(lattice):
    """Test if the stencil is symmetric"""
    assert sum(sum(lattice.e.cpu().numpy())) == pytest.approx(0.0)


def test_weights(lattice):
    """Test if the sum of all weights equals one."""
    assert sum(lattice.w.cpu().numpy()) == pytest.approx(1.0)


def test_first_zero(lattice):
    """Test that the zeroth velocity is 0."""
    assert lattice.stencil.e[0] == pytest.approx(np.zeros_like(lattice.stencil.e[0]))
