
import pytest
import numpy as np
from lettuce.symmetry import *
from lettuce import D1Q3, D2Q9, D3Q19, D3Q27, Lattice


def test_four_rotations(stencil):
    for i in range(stencil.D()):
        for j in range(i+1, stencil.D()):
            assert are_symmetries_equal(
                ChainedSymmetry(*([Rotation90(i, j)]*4)),
                Identity(),
                stencil=stencil
            )
            assert not are_symmetries_equal(
                ChainedSymmetry(*([Rotation90(i, j)]*3)),
                Identity(),
                stencil=stencil
            )


def test_two_reflections(stencil):
    for i in range(stencil.D()):
        assert are_symmetries_equal(
            ChainedSymmetry(*([Reflection(i)]*2)),
            Identity(),
            stencil=stencil
        )
        assert not are_symmetries_equal(
            ChainedSymmetry(*([Reflection(i)]*3)),
            Identity(),
            stencil=stencil
        )


def test_two_reflections(stencil):
    for i in range(stencil.D()):
        assert are_symmetries_equal(
            ChainedSymmetry(*([Reflection(i)]*2)),
            Identity(),
            stencil=stencil
        )
        assert not are_symmetries_equal(
            ChainedSymmetry(*([Reflection(i)]*3)),
            Identity(),
            stencil=stencil
        )


def test_reflection_by_rotations(stencil):
    for i in range(stencil.D()):
        for j in range(i+1, stencil.D()):
            assert are_symmetries_equal(
                ChainedSymmetry(Rotation90(i, j), Rotation90(i, j)),
                ChainedSymmetry(Reflection(i), Reflection(j)),
                stencil=stencil
            )


def test_inverse(stencil):
    for i in range(stencil.D()):
        for j in range(i+1, stencil.D()):
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


@pytest.fixture(scope="module")
def symmetry_group(stencil):
    group = SymmetryGroup(stencil)
    return group


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


def test_feq_equivariance(symmetry_group, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    feq = lambda f: lattice.equilibrium(lattice.rho(f), lattice.u(f))
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    for g in symmetry_group:
        assert np.allclose(
            feq(f[g.permutation(symmetry_group.stencil)]),
            feq(f)[g.permutation(symmetry_group.stencil)],
        )
