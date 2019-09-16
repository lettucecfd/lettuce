
import pytest
from lettuce.moments import *
from lettuce.stencils import *
from lettuce.lattices import Lattice


def test_moments_density_array(stencil):
    rho_tensor = moment_tensor(stencil.e, np.array([0]*stencil.D()))
    assert rho_tensor == pytest.approx(np.ones((stencil.Q())))


def test_more_moments_density_array(stencil):
    rho_tensor = moment_tensor(stencil.e, np.array([[0]*stencil.D()]))
    assert rho_tensor == pytest.approx(np.ones((1,stencil.Q())))


def test_moments_density_tensor(lattice):
    rho_tensor = moment_tensor(lattice.e, lattice.convert_to_tensor(([0]*lattice.D)))
    assert rho_tensor.shape == (lattice.Q,)
    assert rho_tensor.cpu().numpy() == pytest.approx(np.ones((lattice.Q)))


def test_more_moments_density_tensor(lattice):
    rho_tensor = moment_tensor(lattice.e, lattice.convert_to_tensor(([[0]*lattice.D])))
    assert rho_tensor.shape == (1,lattice.Q)
    assert rho_tensor.cpu().numpy() == pytest.approx(np.ones((1,lattice.Q)))


@pytest.mark.parametrize("MomentSet", (D2Q9Dellar, D2Q9Lallemand))
def test_conserved_moments_d2q9(MomentSet):
    multiindices = np.array([
        [0, 0], [1,0], [0,1]
    ])
    m = moment_tensor(D2Q9.e, multiindices)
    assert m == pytest.approx(MomentSet.matrix[:3,:])


def test_inverse_transform(f_transform):
    f, transform = f_transform
    lattice = transform.lattice
    retransformed = lattice.convert_to_numpy(transform.inverse_transform(transform.transform(f)))
    original = lattice.convert_to_numpy(f)
    assert retransformed == pytest.approx(original, abs=1e-6)


def test_getitem(dtype_device):
    dtype, device = dtype_device
    moments = D2Q9Lallemand(Lattice(D2Q9, device, dtype))
    assert moments["jx", "jy"] == [1,2]
    assert moments["rho"] == [0]
