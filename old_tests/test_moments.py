import numpy as np
import pytest
from lettuce.util.moments import *
from lettuce._stencil import *
from lettuce.lattices import Lattice


@pytest.mark.parametrize("MomentSet", (D2Q9Dellar, D2Q9Lallemand, D3Q27Hermite))
def test_orthogonality(dtype_device, MomentSet):
    dtype, device = dtype_device
    lattice = Lattice(MomentSet.supported_stencils[0], device, dtype)
    moments = MomentSet(lattice)
    M = Lattice.convert_to_numpy(moments.matrix)
    if MomentSet == D2Q9Lallemand:
        Md = np.round(M @ M.T, 4)
    else:
        Md = np.round(M @ np.diag(lattice.stencil.w) @ M.T, 4)
    assert np.where(np.diag(np.ones(lattice.stencil.Q())), Md != 0.0, Md == 0.0).all()
