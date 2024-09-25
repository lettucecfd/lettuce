from lettuce._flow import pressure_poisson
from tests.conftest import *


def test_initialize_pressure(fix_dtype):
    context = Context(device='cpu', dtype=fix_dtype, use_native=False)
    flow = TaylorGreenVortex(context=context,
                             resolution=[32] * 2,
                             reynolds_number=1000,
                             mach_number=0.05,
                             initialize_fneq=False)
    """First part: Testing initialize_pressure against analytic solution of 
    TGV."""
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
    # feedback on deviation from analytic solution
    print()
    print("Average absolute deviation of zero-density from analytic "
          "solution: ", (p_before - p0).abs().mean().item())
    print("Average absolute deviation of naive poisson pressure from analytic "
          "solution: ", (p_after - p0).abs().mean().item())
    print("Relative improvement in this case (not representative): ",
          (p_after - p0).abs().mean().item()
          / (p_before - p0).abs().mean().item())
    print("Maximum deviation of poisson pressure from analytic solution: ",
          (p_after - p0).abs().max().item())

    # assert that pressure is converged up to 0.02 (max p)
    p0, p_before, p_after = [context.convert_to_ndarray(_)
                             for _ in [p0, p_before, p_after]]
    assert p_after == pytest.approx(p0, rel=0.0, abs=2e-2)


    """Second part: Testing initialize_pressure to not break if initialized 
    with proper pressure."""
    # getting analytical p and u in LU
    p0, u0 = flow.analytic_solution(t=0)
    u_lu = flow.units.convert_velocity_to_lu(u0)
    rho_lu = flow.units.convert_pressure_pu_to_density_lu(p0)

    # initializing with analytic pressure solution
    f_before = flow.equilibrium(flow, rho=rho_lu, u=u_lu)
    p_before = flow.units.convert_density_lu_to_pressure_pu(flow.rho(f_before))

    # getting poisson calculation for 'better' f's
    rho_poisson = pressure_poisson(flow.units,
                                   u_lu,
                                   rho_lu,
                                   tol_abs=1e-6,
                                   max_num_steps=1000
                                   )
    f_after = flow.equilibrium(flow, rho=rho_poisson, u=u_lu)
    p_after = flow.units.convert_density_lu_to_pressure_pu(flow.rho(f_after))

    # feedback on deviation from analytic solution
    print()
    print("Average absolute deviation of smarter poisson pressure from "
          "analytic solution: ", (p_after - p0).abs().mean().item())
    print("Maximum deviation of smarter poisson pressure from analytic "
          "solution: ", (p_after - p0).abs().max().item())
    print("Average absolute deviation 'improvement' (negative -> worse "
          "solution): ",
          (p_before - p0).abs().mean().item()
          - (p_after - p0).abs().mean().item())

    # assert that equilibrium pressure is similar to analytic solution
    p0, p_before, p_after = [context.convert_to_ndarray(_)
                             for _ in [p0, p_before, p_after]]
    atol = 1e-10 if fix_dtype is torch.float64 else 1e-4
    assert p_before == pytest.approx(p0, rel=0.0, abs=atol)
    # assert that initialized pressure is similar to analytic solution to 0.02
    assert p_after == pytest.approx(p0, rel=0.0, abs=2e-2)
