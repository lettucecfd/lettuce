from lettuce._flow import pressure_poisson
from tests.conftest import *


def test_initialize_pressure(fix_configuration):
    device, dtype, use_native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=use_native)
    flow = TaylorGreenVortex(context=context,
                             resolution=[32] * 2,
                             reynolds_number=10,
                             mach_number=0.05,
                             initialize_fneq=False)
    # getting analytical p and u
    p0, u0 = flow.analytic_solution(t=0)
    # get TGV velocity field
    u_lu = flow.units.convert_velocity_to_lu(u0)
    # manually setting rho=1 (p=0)
    rho_lu = torch.ones_like(p0)

    # initializing as if not knowing analytic pressure solution
    f_before = flow.equilibrium(flow, rho=rho_lu, u=u_lu)
    p_before = flow.units.convert_density_lu_to_pressure_pu(flow.rho(f_before))

    # getting poisson calculation for better f's
    rho_poisson = pressure_poisson(flow.units,
                                   u_lu,
                                   rho_lu,
                                   tol_abs=1e-6,
                                   max_num_steps=1000
                                   )
    f_after = flow.equilibrium(flow, rho=rho_poisson, u=u_lu)
    p_after = flow.units.convert_density_lu_to_pressure_pu(flow.rho(f_after))

    # assert that pressure is much closer to analytic solution
    assert (p_after - p0).abs().sum() < 5e-2 * (p_before - p0).abs().sum()

    # translating to numpy
    p0, p_before, p_after = [context.convert_to_ndarray(_)
                             for _ in [p0, p_before, p_after]]
    # assert that pressure is converged up to 0.05 (max p)
    assert p_after == pytest.approx(p0, rel=0.0, abs=5e-2)
