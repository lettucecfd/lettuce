import warnings
from copy import deepcopy

from tests.simulation.test_initialize_fneq import initialize_pressure
from lettuce.util.moments import get_default_moment_transform
from tests.common import *


def initialize(simulation: 'Simulation',
               max_num_steps=500,
               tol_pressure=0.001):
    """
    Iterative initialization to get moments consistent with the initial
    velocity.

    Using the initialization does not better TGV convergence.
    Maybe use a better scheme?
    """
    warnings.warn(
        "Iterative initialization does not work well and solutions may "
        "diverge. Use with care. Use initialize_f_neq instead.",
        ExperimentalWarning)
    transform = get_default_moment_transform(simulation.flow.stencil.__class__,
                                             simulation.flow.context)
    collision = BGKInitialization(simulation.flow, transform)
    p_old = 0
    for i in range(max_num_steps):
        simulation._collide_and_stream()
        p = simulation.flow.p_pu
        if (torch.max(torch.abs(p - p_old))) < tol_pressure:
            break
        p_old = deepcopy(p)
    return max_num_steps - 1


@pytest.mark.parametrize("use_jacobi", [True, False])
def test_initialization(fix_configuration, use_jacobi):
    pytest.skip("TODO (@PhiSpel): Not working yet")
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
    u0_lu = flow.units.convert_velocity_to_lu(u)
    u0_lu = context.convert_to_tensor(u0_lu)

    p0_pu = torch.zeros_like(p)
    rho0_lu = flow.units.convert_pressure_pu_to_density_lu(p0_pu)
    rho0_lu = context.convert_to_tensor(rho0_lu)
    flow.f = flow.equilibrium(flow=flow, rho=rho0_lu, u=u0_lu)

    if use_jacobi:
        initialize_pressure(flow, 1000, 1e-6)
        num_iterations = 0
    else:
        num_iterations = initialize(simulation, 500, 1e-3)

    piter = flow.p_pu
    print(num_iterations)

    # assert that pressure is converged up to 0.05 (max p
    assert (context.convert_to_ndarray(piter) ==
            pytest.approx(context.convert_to_ndarray(p), rel=0.0, abs=5e-2))
    assert num_iterations < 500
