from lettuce.util.moments import moment_tensor
from tests.conftest import *


def test_moments_density_array(fix_stencil):
    rho_tensor = moment_tensor(fix_stencil.e, np.array([0] * fix_stencil.d))
    assert rho_tensor == pytest.approx(np.ones(fix_stencil.q))


def test_more_moments_density_array(fix_stencil):
    rho_tensor = moment_tensor(fix_stencil.e, np.array([[0] * fix_stencil.d]))
    assert rho_tensor == pytest.approx(np.ones((1, fix_stencil.q)))


def test_moments_density_tensor(fix_stencil, fix_device):
    context = Context(fix_device)
    rho_tensor = moment_tensor(context.convert_to_tensor(fix_stencil.e),
                               context.convert_to_tensor(([0]
                                                          * fix_stencil.d)))
    assert rho_tensor.shape == (fix_stencil.q,)
    assert (context.convert_to_ndarray(rho_tensor)
            == pytest.approx(np.ones((fix_stencil.q))))


def test_more_moments_density_tensor(fix_stencil, fix_device):
    context = Context(fix_device)
    rho_tensor = moment_tensor(context.convert_to_tensor(fix_stencil.e),
                               context.convert_to_tensor(([[0]
                                                           * fix_stencil.d])))
    assert rho_tensor.shape == (1, fix_stencil.q)
    assert (context.convert_to_ndarray(rho_tensor)
            == pytest.approx(np.ones((1, fix_stencil.q))))
