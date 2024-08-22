from lettuce.util.moments import D2Q9Lallemand, D2Q9Dellar
from tests.conftest import *


@pytest.mark.parametrize("Transform", [D2Q9Lallemand, D2Q9Dellar])
def test_collision_fixpoint_2x_MRT(Transform, fix_configuration):
    # TODO: Migrate lattice methods to be independend of Flow or pass Flow
    #  to Transform
    pytest.skip("Transform() currently still uses Lattice methods.")
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    np.random.seed(1)  # arbitrary, but deterministic
    stencil = D2Q9()
    f = context.convert_to_tensor(np.random.random([stencil.q] + [3] *
                                                   stencil.d))
    f_old = copy(f)
    flow = DummyFlow(context, 1)
    collision = MRTCollision(Transform(stencil),
                             [0.5] * 9,
                             context=context)
    f = collision(collision(flow))
    assert f.cpu().numpy() == pytest.approx(f_old.cpu().numpy(), abs=1e-5)
