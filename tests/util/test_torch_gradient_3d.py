from tests.common import *


@pytest.mark.parametrize("order", [2, 4, 6])
def test_torch_gradient_3d(order):
    context = Context()
    flow = TaylorGreenVortex(context=context,
                             resolution=[100] * 3,
                             reynolds_number=1,
                             mach_number=0.05)
    x, y, z = [context.convert_to_ndarray(x) for x in flow.grid]

    _, u = flow.initial_pu()

    dx = flow.units.convert_length_to_pu(1.0)
    u0_grad = torch_gradient(u[0],
                             dx=dx,
                             order=order)

    u = context.convert_to_ndarray(u)
    u0_grad = context.convert_to_ndarray(u0_grad)

    u0_grad_np = np.array(np.gradient(u[0], dx))
    u0_grad_analytic = np.array([
        np.cos(x) * np.cos(y) * np.cos(z),
        np.sin(x) * np.sin(y) * (-np.cos(z)),
        np.sin(x) * (-np.cos(y)) * np.sin(z)
    ])

    assert np.allclose(u0_grad_analytic,
                       u0_grad, rtol=0.0, atol=1e-3)
    assert np.allclose(u0_grad_np[:, 2:-2, 2:-2, 2:-2],
                       u0_grad[:, 2:-2, 2:-2, 2:-2], rtol=0.0, atol=1e-3)
