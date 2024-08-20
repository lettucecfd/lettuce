from tests.common import *

@pytest.mark.parametrize("use_jacobi", [True, False])
def test_initialization(fix_configuration, use_jacobi):
    device, dtype, use_native = fix_configuration
    context = Context(device, dtype, use_native)
    flow = TaylorGreenVortex(context=context,
                             resolution=[24] * 2,
                             reynolds_number=10,
                             mach_number=0.05)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = Simulation(flow, collision, [])
    # set initial pressure to 0 everywhere
    p, u = flow.initial_pu()

    u0 = context.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
    rho0 = context.convert_to_tensor(np.ones_like(u0[0, ...].cpu()))
    flow.f = flow.equilibrium(flow=flow, rho=rho0, u=u0)

    flow.initialize()

    piter = flow.p_pu

    # assert that pressure is converged up to 0.05 (max p
    assert (context.convert_to_ndarray(piter) ==
            pytest.approx(context.convert_to_ndarray(p), rel=0.0, abs=5e-2))