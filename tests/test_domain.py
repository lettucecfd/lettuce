

import pytest
import torch
from torchdd import BoxDomain

@pytest.mark.parametrize("endpoint", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n_cells", [1, 4])
def test_grid_cubic(dim, n_cells, endpoint, ctx):
    domain = BoxDomain(
        lower=torch.zeros(dim, **ctx),
        upper=torch.ones(dim, **ctx),
        resolution=torch.Size([n_cells] * dim),
        endpoint=endpoint,
        cubic_cells=True
    )
    assert domain.n_cells == n_cells**dim
    assert domain.n_points == (n_cells + 1)**dim
    grid = domain.grid()
    assert len(grid) == dim
    for g in grid:
        assert g.shape == tuple([n_cells + 1] * dim)
        assert g.shape == domain.shape
        if endpoint == True:
            assert g.max().item() == pytest.approx(1.0)
        if endpoint == False:
            assert g.max().item() == pytest.approx(n_cells/(n_cells+1))
        assert g.min().item() == pytest.approx(0.0)

    domain.resolution = [2] * dim

    if dim > 1:
        with pytest.raises(ValueError, match="Quad domain with domain lengths"):
            domain.resolution = [2] * (dim-1) + [3]


def test_cubic_fail(ctx):
    with pytest.raises(ValueError, match="Quad domain with domain lengths"):
        BoxDomain(
            lower=torch.zeros(2, **ctx),
            upper=torch.ones(2, **ctx),
            resolution=torch.Size([2, 3]),
            cubic_cells=True
        )


def test_noncubic(ctx):
    domain = BoxDomain(
        lower=torch.zeros(2, **ctx),
        upper=torch.ones(2, **ctx),
        resolution=torch.Size([2, 3]),
        cubic_cells=False
    )
    assert domain.resolution == (2, 3)
    assert domain.shape == (3, 4)
    domain.shape = (5, 6)
    assert domain.resolution == (4, 5)

    grid = domain.grid(as_numpy=True)
    for g in grid:
        assert g.shape == domain.shape


# def test_contains(ctx):
#     domain = BoxDomain(
#         lower=torch.zeros(2, **ctx),
#         upper=torch.ones(2, **ctx),
#         resolution=torch.Size([2, 3]),
#         cubic_cells=False
#     )
#     assert 0.5 * torch.ones(2, **ctx) in domain
#     assert (torch.rand(4, 2, **ctx) in domain)
#     x = torch.rand(4, 2, **ctx)
#     x[2:, :] += 1.0
#     contains = domain.contains(x)
#     assert (contains == torch.Tensor([True, True, False, False]).to(contains)).all()

@pytest.mark.parametrize("endpoint", [True, False])
def test_coarsen_refine(endpoint, ctx):
    domain = BoxDomain(
        lower=torch.zeros(3, **ctx),
        upper=torch.ones(3, **ctx),
        resolution=torch.Size([2, 2, 2]),
        endpoint=endpoint,
        cubic_cells=True
    )
    domain.refine(1)
    assert domain.resolution == (4, 4, 4)
    domain.coarsen(2)
    assert domain.resolution == (1, 1, 1)
    with pytest.raises(ValueError, match="BoxDomain with resolution"):
        domain.coarsen(1)


def test_ghost(ctx):
    domain = BoxDomain(
        lower=torch.zeros(3, **ctx),
        upper=2 * torch.ones(3, **ctx),
        resolution=torch.Size([2, 2, 2]),
        cubic_cells=True,
        n_ghost=[[1, 1], [0, 0], [0, 0]]
    )
    assert (domain.n_ghost == torch.tensor([[1, 1], [0, 0], [0, 0]], dtype=torch.int, device=ctx['device'])).all()
    grid = domain.grid(with_ghost=False)
    for g in grid:
        assert g.shape == domain.shape

    grid_with_ghost = domain.grid(with_ghost=True)
    for g in grid_with_ghost:
        assert g.shape == domain.tensor_shape

    with pytest.raises(ValueError, match="n_ghost needs to have shape"):
        BoxDomain(
            lower=torch.zeros(3, **ctx),
            upper=2*torch.ones(3, **ctx),
            resolution=torch.Size([2, 2, 2]),
            cubic_cells=True,
            n_ghost=[[1,1], [0,0]]
        )

@pytest.mark.parametrize("endpoint", [True, False])
def test_split(endpoint, ctx):
    domain = BoxDomain(
        lower=torch.zeros(3, **ctx),
        upper=2*torch.ones(3, **ctx),
        resolution=torch.Size([10, 10, 10]),
        endpoint=endpoint,
        cubic_cells=True
    )
    # with pytest.raises(ValueError, match="Coordinate not contained in box."):
    #     domain.split(2.9)
    with pytest.raises(ValueError, match="too far from closest grid point"):
        domain.split(1.25)
    # if endpoint == True:
    domains = domain.split(1.0, 1.6)
        # for i in domains:
        #     assert i.resolution[0] in [5,5]
