from tests.common import *


@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_fixpoint_2x(Collision,
                               fix_configuration,
                               fix_stencil):
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context=context,
                    resolution=[16] * fix_stencil.d,
                    reynolds_number=100,
                    mach_number=0.1,
                    stencil=fix_stencil)
    collision = Collision(0.5)
    f_old = copy(flow.f)
    flow.f = collision(flow)
    flow.f = collision(flow)
    assert f_old.cpu().numpy() == pytest.approx(flow.f.cpu().numpy(), abs=1e-5)
