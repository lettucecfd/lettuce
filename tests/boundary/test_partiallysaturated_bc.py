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


def test_fullysaturated_like_bounceback(fix_stencil, fix_configuration):
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow1 = TestFlow(context, resolution=fix_stencil.d * [16],
                     reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    flow2 = TestFlow(context, resolution=fix_stencil.d * [16],
                     reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    tau = flow1.units.relaxation_parameter_lu
    mask = context.one_tensor(flow1.resolution, dtype=bool)  # will contain all
    # points
    PS_BC = PartiallySaturatedBC(mask, tau, saturation=1.0)
    f_fullysaturated = PS_BC(flow1)
    BB_BC = BounceBackBoundary(mask)
    f_bounced = BB_BC(flow2)
    assert (f_fullysaturated.cpu().numpy() ==
            pytest.approx(f_bounced.cpu().numpy()))
    


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