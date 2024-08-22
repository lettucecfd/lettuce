from tests.common import *


def test_moment_equilibrium_dellar(fix_device, fix_dtype):
    context = Context(fix_device, fix_dtype)
    stencil = D2Q9()
    moments = D2Q9Dellar(stencil, context)
    flow = TestFlow(context, 10, 1, 0.1, stencil=stencil)
    meq1 = context.convert_to_ndarray(moments.transform(flow.equilibrium(
        flow)))
    meq2 = context.convert_to_ndarray(moments.equilibrium(moments.transform(
        flow.f), flow))
    assert meq1 == pytest.approx(meq2, abs=1e-5)
