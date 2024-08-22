from tests.common import *


def test_moment_equilibrium_lallemand(fix_device, fix_dtype):
    context = Context(fix_device, fix_dtype)
    stencil = D2Q9()
    moments = D2Q9Lallemand(stencil, context)
    flow = TestFlow(context, 10, 1, 0.1, stencil=stencil)
    meq1 = context.convert_to_ndarray(moments.transform(flow.equilibrium(
        flow)))
    meq2 = context.convert_to_ndarray(moments.equilibrium(moments.transform(
        flow.f), flow))
    same_moments = moments["rho", "jx", "jy", "qx", "qy"]
    assert meq1[same_moments] == pytest.approx(meq2[same_moments], abs=1e-5)
