import pytest
import torch
import numpy as np
from lettuce.symmetry import *
from lettuce.stencils import D1Q3, D2Q9, D3Q19, D3Q27
from lettuce.lattices import Lattice
from lettuce.collision import MRTCollision
from lettuce.util import LettuceCollisionNotDefined
from lettuce.moments import DEFAULT_TRANSFORM


def test_four_rotations(stencil):
    for i in range(stencil.D()):
        for j in range(i + 1, stencil.D()):
            assert are_symmetries_equal(
                ChainedSymmetry(*([Rotation90(i, j)] * 4)),
                Identity(),
                stencil=stencil
            )
            assert not are_symmetries_equal(
                ChainedSymmetry(*([Rotation90(i, j)] * 3)),
                Identity(),
                stencil=stencil
            )


def test_two_reflections(stencil):
    for i in range(stencil.D()):
        assert are_symmetries_equal(
            ChainedSymmetry(*([Reflection(i)] * 2)),
            Identity(),
            stencil=stencil
        )
        assert not are_symmetries_equal(
            ChainedSymmetry(*([Reflection(i)] * 3)),
            Identity(),
            stencil=stencil
        )


def test_reflection_by_rotations(stencil):
    for i in range(stencil.D()):
        for j in range(i + 1, stencil.D()):
            assert are_symmetries_equal(
                ChainedSymmetry(Rotation90(i, j), Rotation90(i, j)),
                ChainedSymmetry(Reflection(i), Reflection(j)),
                stencil=stencil
            )


def test_inverse(stencil):
    for i in range(stencil.D()):
        for j in range(i + 1, stencil.D()):
            assert are_symmetries_equal(
                ChainedSymmetry(Rotation90(i, j), InverseSymmetry(Rotation90(i, j))),
                Identity(),
                stencil=stencil
            )
        assert are_symmetries_equal(
            ChainedSymmetry(Reflection(i), InverseSymmetry(Reflection(i))),
            Identity(),
            stencil=stencil
        )


def test_symmetry_group(symmetry_group):
    group = symmetry_group
    n_symmetries = {D1Q3: 2, D2Q9: 8, D3Q19: 48, D3Q27: 48}[group.stencil]
    assert len(group) == n_symmetries
    assert group.permutations.shape == (n_symmetries, group.stencil.Q())

    # assert that it's a group:
    # contains the identity
    assert Identity() in group
    for s1 in group:
        for s2 in group:
            if s1 != s2:
                # elements are unique
                assert not are_symmetries_equal(s1, s2, group.stencil)
        # contains the inverse
        assert InverseSymmetry(s1) in group


def test_permutations(symmetry_group):
    for g in symmetry_group:
        # symmetry operation is equal to the corresponding permutation
        assert np.allclose(
            g.forward(symmetry_group.stencil.e),
            symmetry_group.stencil.e[g.permutation(symmetry_group.stencil), ...]
        )
        # inverse permutation of permutation is identity
        perm1 = g.permutation(symmetry_group.stencil)
        perm2 = InverseSymmetry(g).permutation(symmetry_group.stencil)
        assert np.allclose(perm1[perm2], np.arange(symmetry_group.stencil.Q()))
        assert np.allclose(perm2[perm1], np.arange(symmetry_group.stencil.Q()))


def test_inverse_permutations(symmetry_group):
    for p, pinv in zip(symmetry_group.permutations, symmetry_group.inverse_permutations):
        assert (p[pinv] == np.arange(symmetry_group.stencil.Q())).all()
        assert (pinv[p] == np.arange(symmetry_group.stencil.Q())).all()


def test_moment_representations(symmetry_group):
    try:
        transform = DEFAULT_TRANSFORM[symmetry_group.stencil]
    except KeyError:
        pytest.skip("No default transform for this stencil")
    rep = symmetry_group.moment_representations(transform)
    irep = symmetry_group.inverse_moment_representations(transform)
    # test if this is a representation
    # group op = matrix multiply
    for i, symmetry in enumerate(symmetry_group):
        for j, symmetry2 in enumerate(symmetry_group):
            ji = symmetry_group.index(ChainedSymmetry(symmetry, symmetry2))
            assert np.allclose(rep[j] @ rep[i], rep[ji])
    # inverse group op = inverse matrix
    for forward, inverse in zip(rep, irep):
        assert np.allclose(forward @ inverse, np.eye(symmetry_group.stencil.Q()))
        assert np.allclose(inverse @ forward, np.eye(symmetry_group.stencil.Q()))


def test_feq_equivariance(symmetry_group, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    feq = lambda x: lattice.equilibrium(lattice.rho(x), lattice.u(x))
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    for g in symmetry_group:
        assert torch.allclose(
            feq(f[g.permutation(symmetry_group.stencil)]),
            feq(f)[g.permutation(symmetry_group.stencil)],
        )


def test_collision_equivariance(symmetry_group, dtype_device, Collision):
    """Test whether all collision models obey the lattice symmetries."""
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    try:
        collision = Collision(lattice, 0.51)
    except LettuceCollisionNotDefined:
        pytest.skip()
    f_post = collision(f.clone())
    for permutation in symmetry_group.permutations:
        f_post_after_g = collision(f.clone()[permutation])
        assert torch.allclose(
            f_post_after_g,
            f_post[permutation],
            atol=2e-5 if dtype == torch.float32 else 1e-7
        ), f"{(f_post_after_g - f_post[permutation]).norm()}"


def test_non_equivariant_mrt(dtype_device):
    dtype, device = dtype_device
    stencil = D2Q9
    lattice = Lattice(stencil, dtype=dtype, device=device)
    symmetry_group = SymmetryGroup(D2Q9)
    # non-equivariant choice of relaxation parameters
    collision = MRTCollision(lattice, torch.arange(9.0))
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    f_post = collision(f.clone())
    is_equivariant = True
    for permutation in symmetry_group.permutations:
        f_post_after_g = collision(f.clone()[permutation])
        are_equal = torch.allclose(
            f_post_after_g,
            f_post[permutation],
            atol=2e-5 if dtype == torch.float32 else 1e-7
        )
        if not are_equal:
            is_equivariant = False
    assert not is_equivariant
