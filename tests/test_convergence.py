from tests.common import *


def test_convergence(fix_configuration):
    """Use Taylor Green 2D for convergence test in diffusive scaling."""
    device, dtype, use_native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=use_native)

    error_u_old = None
    error_p_old = None
    factor_u = None
    factor_p = None
    print(("{:>15} " * 6).format("resolution", "error (u)", "order (u)",
                                 "error (p)", "order (p)", "MLUPS"))

    for i in range(4, 9 if dtype==torch.float64 else 7):  # single
        # precission does not converge as far
        resolution = 2 ** i
        mach_number = 8 / resolution

        # Simulation
        flow = TaylorGreenVortex(context, [resolution] * 2,
                                 reynolds_number=10000,
                                 mach_number=mach_number)
        collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
        error_reporter = ErrorReporter(flow.analytic_solution, interval=1,
                                       out=None)

        simulation = Simulation(flow, collision, [error_reporter])

        mlups = simulation(10 * resolution)

        error_u, error_p = np.mean(np.abs(error_reporter.out), axis=0).tolist()
        factor_u = 0 if error_u_old is None else error_u_old / error_u
        factor_p = 0 if error_p_old is None else error_p_old / error_p
        error_u_old = error_u
        error_p_old = error_p

        print(f"{resolution:15} {error_u:15.2e} {factor_u / 2:15.1f} "
              f"{error_p:15.2e} {factor_p / 2:15.1f} {mlups:15.2f}")
    assert factor_u / 2 > (2 - 1e-6), "Velocity convergence failed"
    assert factor_p / 2 > (1 - 1e-6), "Pressure convergence failed"
