import numpy as np
import pytest
from lettuce.util.moments import *
from lettuce._stencil import *
from lettuce.lattices import Lattice


def test_getitem(dtype_device):
    dtype, device = dtype_device
    moments = D2Q9Lallemand(Lattice(D2Q9, device, dtype))
    assert moments["jx", "jy"] == [1, 2]
    assert moments["rho"] == [0]


def test_moment_equilibrium_dellar(dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype)
    moments = D2Q9Dellar(lattice)
    np.random.seed(1)
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    meq1 = lattice.convert_to_numpy(moments.transform(lattice.equilibrium(lattice.rho(f), lattice.u(f))))
    meq2 = lattice.convert_to_numpy(moments.equilibrium(moments.transform(f)))
    assert meq1 == pytest.approx(meq2, abs=1e-5)


def test_moment_equilibrium_lallemand(dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device, dtype)
    moments = D2Q9Lallemand(lattice)
    np.random.seed(1)
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    meq1 = lattice.convert_to_numpy(moments.transform(lattice.equilibrium(lattice.rho(f), lattice.u(f))))
    meq2 = lattice.convert_to_numpy(moments.equilibrium(moments.transform(f)))
    same_moments = moments["rho", "jx", "jy", "qx", "qy"]
    assert meq1[same_moments] == pytest.approx(meq2[same_moments], abs=1e-5)


def test_moment_equilibrium_D3Q27Hermite(dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D3Q27, device, dtype)
    moments = D3Q27Hermite(lattice)
    np.random.seed(1)
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    meq1 = lattice.convert_to_numpy(moments.transform(lattice.equilibrium(lattice.rho(f), lattice.u(f))))
    meq2 = lattice.convert_to_numpy(moments.equilibrium(moments.transform(f)))
    same_moments = moments['rho', 'jx', 'jy', 'jz', 'Pi_xx', 'Pi_xy', 'PI_xz', 'PI_yy', 'PI_yz', 'PI_zz']
    assert meq1[same_moments] == pytest.approx(meq2[same_moments], abs=1e-5)


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
