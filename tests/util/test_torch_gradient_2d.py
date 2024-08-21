from tests.common import *


@pytest.mark.parametrize("order", [2, 4, 6])
def test_torch_gradient_2d(order):
    context = Context()
    flow = TaylorGreenVortex(context=context,
                             resolution=[100] * 2,
                             reynolds_number=1,
                             mach_number=0.05)
    x, y = flow.grid
    x = context.convert_to_ndarray(x)
    y = context.convert_to_ndarray(y)
    _, u = flow.initial_pu()
    dx = flow.units.convert_length_to_pu(1.0)
    u0_grad = torch_gradient(u[0],
                             dx=dx,
                             order=order)
    u = context.convert_to_ndarray(u)
    u0_grad = context.convert_to_ndarray(u0_grad)
    u0_grad_np = np.array(np.gradient(u[0], dx))
    u0_grad_analytic = np.array([
        -np.sin(x) * np.sin(y),
        np.cos(x) * np.cos(y),
    ])
    assert (u0_grad_analytic
            == pytest.approx(u0_grad, rel=0.0, abs=1e-3))
    assert (u0_grad_np[:, 2:-2, 2:-2]
            == pytest.approx(u0_grad[:, 2:-2, 2:-2], rel=0.0, abs=1e-3))
