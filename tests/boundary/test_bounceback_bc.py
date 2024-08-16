from tests.common import *

from copy import copy


def test_bounce_back_boundary(fix_stencil, fix_configuration):
    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)
    flow = TestFlow(context, resolution=fix_stencil.d * [16],
                    reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    f_old = copy(flow.f)
    mask = context.one_tensor(flow.resolution)  # will contain all points
    bounce_back = BounceBackBoundary(mask)
    f_bounced = bounce_back(flow)
    assert (f_old[flow.stencil.opposite].cpu().numpy() ==
            pytest.approx(f_bounced.cpu().numpy()))

