from tests.common import *


@pytest.mark.parametrize("Collision",
                         [BGKCollision, KBCCollision, TRTCollision,
                          RegularizedCollision, SmagorinskyCollision])
def test_collision_conserves_mass(Collision,
                                  fix_configuration,
                                  fix_stencil):
    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)
    flow = TestFlow(context=context,
                    resolution=[16] * fix_stencil.d,
                    reynolds_number=100,
                    mach_number=0.1)

    if Collision == KBCCollision and fix_stencil not in [D2Q9, D3Q27]:
        pytest.skip("KBCCollision only implemented for D2Q9 and D3Q27")
    collision = Collision(tau=0.51)
    f_collided = collision(flow)
    assert (flow.rho().cpu().numpy()
            == pytest.approx(flow.rho(f_collided).cpu().numpy()))

