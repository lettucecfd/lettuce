"""
Tests for streaming operators.
"""

from lettuce import StandardStreaming

import pytest

import copy


def test_standard_streaming_x3(f_lattice):
    """Streaming three times on a 3^D grid gives the original distribution functions."""
    f, lattice = f_lattice
    f_old = copy.copy(f.cpu().numpy())
    streaming = StandardStreaming(lattice)
    f = streaming(streaming(streaming(f)))
    assert f.cpu().numpy() == pytest.approx(f_old)




