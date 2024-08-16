from tests.common import *

@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_fixpoint_2x(Collision, fix_configuration, fix_stencil):
    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)
    flow = TestFlow(context=context,
                    resolution=[16] * fix_stencil.d,
                    reynolds_number=100,
                    mach_number=0.1)
    collision = Collision(0.5)
    f_old = copy(flow.f)
    flow.f = collision(flow)
    flow.f = collision(flow)
    assert f_old.cpu().numpy() == pytest.approx(flow.f.cpu().numpy(), abs=1e-5)
