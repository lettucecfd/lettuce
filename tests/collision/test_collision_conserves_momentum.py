from tests.common import *


def test_collision_conserves_momentum(fix_conserving_collision,
                                      fix_configuration,
                                      fix_stencil):
    if (fix_conserving_collision == KBCCollision and type(fix_stencil) not
            in [D2Q9, D3Q27]):
        pytest.skip("KBCCollision only implemented for D2Q9 and D3Q27")
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    flow = TestFlow(context=context,
                    resolution=[16] * fix_stencil.d,
                    reynolds_number=100,
                    mach_number=0.1,
                    stencil=fix_stencil)
    collision = fix_conserving_collision(tau=0.51)
    f_collided = collision(flow)
    assert (flow.j().cpu().numpy()
            == pytest.approx(flow.j(f_collided).cpu().numpy(), abs=1e-5))
