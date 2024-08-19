from tests.common import *

@pytest.mark.parametrize("stencil2d3d", [D2Q9(), D3Q27()])
def test_divergence(stencil2d3d, fix_configuration):
    device, dtype, use_native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=use_native)
    flow = DecayingTurbulence(context=context,
                              resolution=[50] * stencil2d3d.d,
                              reynolds_number=1,
                              mach_number=0.05,
                              ic_energy=0.5)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = Simulation(flow=flow,
                            collision=collision,
                            reporter=[])
    ekin = (flow.units.convert_incompressible_energy_to_pu(
        torch.sum(flow.incompressible_energy())) *
            flow.units.convert_length_to_pu(1.0) ** stencil2d3d.d)

    u0 = flow.u_pu[0]
    u1 = flow.u_pu[1]
    dx = flow.units.convert_length_to_pu(1.0)
    grad_u0 = torch_gradient(u0, dx=dx, order=6).cpu().numpy()
    grad_u1 = torch_gradient(u1, dx=dx, order=6).cpu().numpy()
    divergence = np.sum(grad_u0[0] + grad_u1[1])

    if stencil2d3d.d == 3:
        u2 = flow.u_pu[2]
        grad_u2 = torch_gradient(u2, dx=dx, order=6).cpu().numpy()
        divergence += np.sum(grad_u2[2])
    assert (flow.ic_energy ==
            pytest.approx(context.convert_to_ndarray(ekin), rel=1))
    assert (0 == pytest.approx(divergence, abs=2e-3))
    