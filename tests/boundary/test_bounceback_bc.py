from tests.conftest import *

from copy import copy


def test_bounce_back_boundary(fix_stencil, fix_configuration):
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context, resolution=fix_stencil.d * [16],
                    reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    f_old = copy(flow.f)
    mask = context.one_tensor(flow.resolution, dtype=bool)  # will contain all
    # points
    bounce_back = BounceBackBoundary(mask)
    f_bounced = bounce_back(flow)
    assert (f_old[flow.stencil.opposite].cpu().numpy() ==
            pytest.approx(f_bounced.cpu().numpy()))


def test_bounce_back_boundary_not_applied_if_mask_empty(fix_stencil,
                                                        fix_configuration):
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context, resolution=fix_stencil.d * [16],
                    reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    f_old = copy(flow.f)
    mask = context.zero_tensor(flow.resolution, dtype=bool)  # will not contain
    # any points
    bounce_back = BounceBackBoundary(mask)
    bounce_back(flow)
    assert (flow.f.cpu().numpy() ==
            pytest.approx(f_old.cpu().numpy()))
