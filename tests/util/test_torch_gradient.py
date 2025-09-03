from tests.conftest import *


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("order", [2, 4, 6])
def test_torch_gradient(dims, order):
    context = Context()
    flow = TaylorGreenVortex(context=context,
                             resolution=[100] * dims,
                             reynolds_number=1,
                             mach_number=0.05)

    _, u = flow.initial_pu()

    dx = 2 * torch.pi / 100
    u0_grad = torch_gradient(u[0],
                             dx=dx,
                             order=order)

    u = context.convert_to_ndarray(u)
    u0_grad = context.convert_to_ndarray(u0_grad)

    u0_grad_np = np.array(np.gradient(u[0], dx))
    if dims == 2:
        x, y = [context.convert_to_ndarray(x) for x in flow.grid]
        u0_grad_analytic = np.array([
            -np.sin(x) * np.sin(y),
            np.cos(x) * np.cos(y),
        ])
    elif dims == 3:
        x, y, z = [context.convert_to_ndarray(x) for x in flow.grid]
        u0_grad_analytic = np.array([
            np.cos(x) * np.cos(y) * np.cos(z),
            np.sin(x) * np.sin(y) * (-np.cos(z)),
            np.sin(x) * (-np.cos(y)) * np.sin(z)
        ])
    else:
        return
    assert np.allclose(u0_grad_analytic,
                       u0_grad, rtol=1e-3, atol=1e-3)
    if dims == 2:
        assert (u0_grad_np[:, 2:-2, 2:-2]
                == pytest.approx(u0_grad[:, 2:-2, 2:-2],
                                 rel=1e-3, abs=1e-3))
    if dims == 3:
        assert (u0_grad_np[:, 2:-2, 2:-2, 2:-2]
                == pytest.approx(u0_grad[:, 2:-2, 2:-2, 2:-2],
                                 rel=1e-3, abs=1e-3))
