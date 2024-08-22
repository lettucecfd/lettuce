"""Testing that initializing fneq improves TGV solution"""
import pytest

from tests.conftest import *


@pytest.mark.parametrize("Case", [DecayingTurbulence,
                                  TaylorGreenVortex,
                                  DoublyPeriodicShear2D])
def test_initialize_fneq(fix_configuration, fix_stencil, Case):
    # fixture setup
    if fix_stencil.d == 1:
        pytest.skip("Testflows not working for 1D")
    if Case is DoublyPeriodicShear2D and fix_stencil.d != 2:
        pytest.skip("DoublyPeriodicShear2D only working for 2D")
    device, dtype, use_native = fix_configuration
    # if dtype == torch.float32:
    #     pytest.skip("TGV is not accurate enough for single precision.")
    context = Context(device, dtype, use_native)

    # setting up flows with and without fneq
    if Case is DecayingTurbulence:
        randseed = np.random.randint(1)
        flow_neq = Case(context=context, resolution=32, reynolds_number=1000,
                        mach_number=0.1, stencil=fix_stencil,
                        initialize_pressure=False,
                        randseed=randseed)
        flow_eq = Case(context=context, resolution=32, reynolds_number=1000,
                       mach_number=0.1, stencil=fix_stencil,
                       initialize_pressure=False,
                       initialize_fneq=False,
                       randseed=randseed)
    else:
        flow_neq = Case(context=context, resolution=32, reynolds_number=1000,
                    mach_number=0.1, stencil=fix_stencil)
        flow_eq = Case(context=context, resolution=32, reynolds_number=1000,
                       mach_number=0.1, stencil=fix_stencil,
                       initialize_fneq=False)

    # initializing with and without fneq
    flow_eq.initialize()
    flow_neq.initialize()

    # comparing densitiy, velocity, and kinetic energy with and without fneq
    # (should be equal)
    rho_eq = flow_eq.rho()
    u_eq = flow_eq.u()
    ke_eq = flow_eq.incompressible_energy()

    rho_neq = flow_neq.rho()
    u_neq = flow_neq.u()
    ke_neq = flow_neq.incompressible_energy()

    print(u_eq)
    print(u_neq)

    tol = 1e-6
    assert (context.convert_to_ndarray(rho_neq)
            == pytest.approx(context.convert_to_ndarray(rho_eq),
                             rel=0.0,
                             abs=tol))
    assert (context.convert_to_ndarray(u_neq)
            == pytest.approx(context.convert_to_ndarray(u_eq),
                             rel=0.0,
                             abs=tol))
    assert (context.convert_to_ndarray(ke_neq)
            == pytest.approx(context.convert_to_ndarray(ke_eq),
                             rel=0.0,
                             abs=tol))

    # comparing to analytic solution of TGV
    if Case is TaylorGreenVortex and fix_stencil.d == 2:
        collision = BGKCollision(tau=flow_neq.units.relaxation_parameter_lu)

        error_reporter_neq = ErrorReporter(flow_neq.analytic_solution,
                                           interval=1,
                                           out=None)
        simulation_neq = Simulation(flow_neq, collision, [error_reporter_neq])

        error_reporter_eq = ErrorReporter(flow_eq.analytic_solution,
                                          interval=1,
                                          out=None)
        simulation_eq = Simulation(flow_eq, collision, [error_reporter_eq])

        simulation_neq(10)
        simulation_eq(10)

        error_u, error_p = np.mean(np.abs(error_reporter_neq.out),
                                   axis=0).tolist()
        error_u_eq, error_p_eq = np.mean(np.abs(error_reporter_eq.out),
                                         axis=0).tolist()

        assert error_u < error_u_eq
