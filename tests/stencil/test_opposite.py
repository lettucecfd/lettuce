from tests.common import *

def test_opposite(fix_stencil):
    """Test if the opposite field holds the index of the opposite direction."""
    context = Context()
    torch_stencil = TorchStencil(fix_stencil, context)
    assert torch.isclose(torch_stencil.e[fix_stencil.opposite],
                         -torch_stencil.e).all()
