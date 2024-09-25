from tests.conftest import *


def test_moment_equilibrium_D3Q27Hermite(fix_device, fix_dtype):
    context = Context(fix_device, fix_dtype)
    stencil = D3Q27()
    moments = D3Q27Hermite(stencil, context)
    flow = TestFlow(context, 10, 1, 0.1, stencil=stencil)
    meq1 = context.convert_to_ndarray(moments.transform(flow.equilibrium(
        flow)))
    meq2 = context.convert_to_ndarray(moments.equilibrium(moments.transform(
        flow.f), flow))
    same_moments = moments['rho', 'jx', 'jy', 'jz', 'Pi_xx', 'Pi_xy', 'PI_xz', 'PI_yy', 'PI_yz', 'PI_zz']
    assert meq1[same_moments] == pytest.approx(meq2[same_moments], abs=1e-5)
