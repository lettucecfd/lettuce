
import pytest


def test_opposite(lattice):
    """Test if the opposite field holds the index of the opposite direction."""
    assert lattice.e[lattice.stencil.opposite].cpu().numpy() == pytest.approx(-lattice.e.cpu().numpy())
