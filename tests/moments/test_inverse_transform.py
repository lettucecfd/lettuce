from tests.common import *


def test_inverse_transform(fix_transform):
    Transform, Stencil = fix_transform
    stencil = Stencil()
    context = Context()
    torch_stencil = TorchStencil(stencil, context)
    transform = Transform(torch_stencil, context)
    f = context.convert_to_tensor(
        np.random.random([stencil.q] + [3] * stencil.d))
    original = context.convert_to_ndarray(f)
    retransformed = context.convert_to_ndarray(
        transform.inverse_transform(transform.transform(f)))
    assert retransformed == pytest.approx(original, abs=1e-5)
