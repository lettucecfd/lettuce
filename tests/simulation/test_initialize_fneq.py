from tests.common import *


def initialize_pressure(flow: 'Flow', max_num_steps=100000, tol_pressure=1e-6):
    """Reinitialize equilibrium distributions with pressure obtained by a
    Jacobi solver. Note that this method has to be called before
    initialize_f_neq.
    """
    u = flow.u()
    rho = pressure_poisson(
        flow.units,
        u,
        flow.rho(),
        tol_abs=tol_pressure,
        max_num_steps=max_num_steps
    )
    return flow.equilibrium(flow=flow, rho=rho, u=u)


def initialize_f_neq(flow: 'Flow'):
    """Initialize the distribution function values. The f^(1) contributions are
    approximated by finite differences. See Krüger et al. (2017).
    """
    rho = flow.rho()
    u = flow.u()

    grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
    grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
    S = torch.cat([grad_u0, grad_u1])

    if flow.stencil.d == 3:
        grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
        S = torch.cat([S, grad_u2])

    Pi_1 = (1.0 * flow.units.relaxation_parameter_lu * rho * S
            / flow.stencil.cs ** 2)
    Q = (torch.einsum('ia,ib->iab', [flow.stencil.e, flow.stencil.e])
         - torch.eye(flow.torch_stencil.d) * flow.stencil.cs ** 2)
    Pi_1_Q = flow.einsum('ab,iab->i', [Pi_1, Q])
    fneq = flow.einsum('i,i->i', [flow.stencil.w, Pi_1_Q])

    feq = flow.equilibrium(flow, rho, u)

    return feq - fneq


@pytest.mark.parametrize("Case", [TaylorGreenVortex, DecayingTurbulence])
@pytest.mark.parametrize("Stencils", [D2Q9, D3Q19, D3Q27])
def test_initialize_fneq(Case, Stencils, fix_configuration):
    # TODO: Should we re-implement pressure_poisson initialize_f_neq?!
    device, dtype, use_native = fix_configuration
    context = Context(device, dtype, use_native)
    flow = Case(context=context,
                resolution=32,
                reynolds_number=1000,
                mach_number=0.1,
                stencil=Stencils())
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation_neq = Simulation(flow, collision, [])

    pre_rho = flow.rho(simulation_neq.flow.f)
    pre_u = flow.u(simulation_neq.flow.f)
    pre_ke = flow.incompressible_energy(simulation_neq.flow.f)

    initialize_pressure(flow)

    post_rho = flow.rho(simulation_neq.flow.f)
    post_u = flow.u(simulation_neq.flow.f)
    post_ke = flow.incompressible_energy(simulation_neq.flow.f)
    tol = 1e-6
    assert torch.allclose(pre_rho, post_rho, rtol=0.0, atol=tol)
    assert torch.allclose(pre_u, post_u, rtol=0.0, atol=tol)
    assert torch.allclose(pre_ke, post_ke, rtol=0.0, atol=tol)

    if Case is TaylorGreenVortex and Stencils().d == 2:
        error_reporter_neq = ErrorReporter(flow.analytic_solution, interval=1,
                                           out=None)
        error_reporter_eq = ErrorReporter(flow.analytic_solution,
                                          interval=1, out=None)
        simulation_eq = Simulation(flow, collision, [])
        simulation_neq.reporter.append(error_reporter_neq)
        simulation_eq.reporter.append(error_reporter_eq)

        simulation_neq(10)
        simulation_eq(10)
        error_u, error_p = np.mean(np.abs(error_reporter_neq.out),
                                   axis=0).tolist()
        error_u_eq, error_p_eq = np.mean(np.abs(error_reporter_eq.out),
                                         axis=0).tolist()

        assert error_u < error_u_eq
