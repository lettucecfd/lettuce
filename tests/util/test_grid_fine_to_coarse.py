from tests.common import *


@pytest.mark.parametrize("dims", [2, 3])
def test_grid_fine_to_coarse(dims):
    context = Context(device='cpu', dtype=torch.float64)
    flow_f = TaylorGreenVortex(context, [40] * dims, 1600, 0.15)

    flow_c = TaylorGreenVortex(context, [20] * dims, 1600, 0.15)

    f_c = grid_fine_to_coarse(flow_f,
                              flow_f.f,
                              flow_f.units.relaxation_parameter_lu,
                              flow_c.units.relaxation_parameter_lu)

    p_init, u_init = flow_c.initial_pu()
    rho_init = flow_c.units.convert_pressure_pu_to_density_lu(p_init)
    u_init = flow_c.units.convert_velocity_to_lu(u_init)
    shear_c_init = flow_c.shear_tensor(flow_c.f)
    shear_c = flow_c.shear_tensor(f_c)

    assert torch.isclose(flow_c.u(f_c), u_init).all()
    assert torch.isclose(flow_c.rho(f_c), rho_init).all()
    assert torch.isclose(f_c, flow_c.f).all()
    assert torch.isclose(shear_c_init, shear_c).all()
