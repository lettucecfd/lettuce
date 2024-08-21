from tests.common import *


def test_orthogonality(fix_device, fix_dtype, fix_transform):
    Transform, Stencil = fix_transform
    stencil = Stencil()
    if stencil.d == 1:
        pytest.skip("No othogonality for 1D")
    context = Context(fix_device, fix_dtype)
    tranform = Transform(stencil, context)
    M = context.convert_to_ndarray(tranform.matrix)
    if Transform is D2Q9Lallemand:
        Md = np.round(M @ M.T, 4)
    else:
        Md = np.round(M @ np.diag(stencil.w) @ M.T, 4)
    assert np.where(np.diag(np.ones(stencil.q)), Md != 0.0, Md == 0.0).all()
