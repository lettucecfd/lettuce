from tests.conftest import *

from copy import copy


def test_partiallysaturated_boundary(fix_stencil, fix_configuration):
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context, resolution=fix_stencil.d * [16],
                    reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    tau = flow.units.relaxation_parameter_lu
    collision = BGKCollision(tau)
    mask = context.one_tensor(flow.resolution, dtype=bool)  # will contain all
    # points
    flow.boundaries = [PartiallySaturatedBC(mask,
                                            flow.units.relaxation_parameter_lu,
                                            saturation=0.5)]
    simulation = Simulation(flow, collision, [])
    simulation(2)


def test_fullysaturated_like_neq_bounceback(fix_stencil, fix_configuration):
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    if dtype == torch.float32:
        pytest.skip("This test does not work for single precision.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context, resolution=fix_stencil.d * [16],
                    reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    flow.f = torch.rand_like(flow.f, dtype=dtype)*flow.f
    mask = context.one_tensor(flow.resolution, dtype=bool)  # will contain all
    # points

    # create neq part
    tau = flow.units.relaxation_parameter_lu
    collision = BGKCollision(tau)
    flow.f = collision(flow)
    fneq_bounceback = (flow.equilibrium(flow) - flow.f)[flow.stencil.opposite]

    PS_BC = PartiallySaturatedBC(mask, tau, saturation=1.0)
    f_fullysaturated = PS_BC(flow)

    flow2 = copy(flow)
    flow2.f = f_fullysaturated
    fneq_fullysaturated = flow2.equilibrium(flow2) - flow2.f
    
    assert (fneq_bounceback.cpu().numpy() ==
            pytest.approx(fneq_fullysaturated.cpu().numpy()))
    


def test_partiallysaturated_boundary_not_applied_if_mask_empty(fix_stencil,
                                                        fix_configuration):
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context, resolution=fix_stencil.d * [16],
                    reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    tau = flow.units.relaxation_parameter_lu
    mask = context.zero_tensor(flow.resolution, dtype=bool)  # will not contain
    # any points
    f_old = copy(flow.f)
    PS_BC = PartiallySaturatedBC(mask, tau, saturation=0.5)
    f_bounced = PS_BC(flow)
    assert (f_bounced.cpu().numpy() ==
            pytest.approx(f_old.cpu().numpy()))