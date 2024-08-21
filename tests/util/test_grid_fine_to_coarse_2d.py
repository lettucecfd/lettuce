from tests.common import *


def test_grid_fine_to_coarse_2d():
    context = Context(dtype=torch.float64)
    flow_f = TaylorGreenVortex(context, [40] * 2, 1600, 0.15)
    collision_f = BGKCollision(tau=flow_f.units.relaxation_parameter_lu)
    # sim_f = Simulation(flow_f, collision_f, [])

    flow_c = TaylorGreenVortex(context, [20] * 2, 1600, 0.15)
    collision_c = BGKCollision(tau=flow_c.units.relaxation_parameter_lu)
    # sim_c = Simulation(flow_c, collision_c, [])

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
