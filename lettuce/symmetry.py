"""Lattice Symmetries"""

import numpy as np

__all__ = [
    "is_symmetry", "are_symmetries_equal", "Symmetry",
    "SymmetryGroup", "InverseSymmetry", "ChainedSymmetry", "Identity",
    "Rotation90", "Reflection"
]


def is_symmetry(operation, stencil):
    "whether the operation leaves the stencil invariant"
    original_e = set(tuple(e) for e in stencil.e)
    new_e = set(tuple(operation.forward(e)) for e in stencil.e)
    reverse_e = set(tuple(operation.inverse(e)) for e in stencil.e)
    return original_e == new_e and original_e == reverse_e


def are_symmetries_equal(symmetry1, symmetry2, stencil):
    return np.allclose(symmetry1.forward(stencil.e), symmetry2.forward(stencil.e))


class Symmetry:
    """Abstract base class for symmetry operations."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return NotImplemented

    def inverse(self, x):
        return NotImplemented

    def permutation(self, stencil):
        assert is_symmetry(self, stencil)
        other = self.forward(stencil.e)
        return np.concatenate([np.where((ei == stencil.e).all(axis=-1))[0] for ei in other])


class ChainedSymmetry(Symmetry):
    """Stitch multiple symmetries together."""

    def __init__(self, *symmetries):
        super().__init__()
        self.symmetries = []
        # unfold chains
        for i, symmetry in enumerate(symmetries):
            if isinstance(symmetry, ChainedSymmetry):
                self.symmetries.extend([*symmetry])
            else:
                self.symmetries.append(symmetry)
        self.symmetries = tuple(self.symmetries)

    def forward(self, x):
        for s in self.symmetries:
            x = s.forward(x)
        return x

    def inverse(self, x):
        for s in reversed(self.symmetries):
            x = s.inverse(x)
        return x

    def __iter__(self):
        return self.symmetries.__iter__()

    def __len__(self):
        return self.symmetries.__len__()

    def __repr__(self):
        return f"<Chain: {[repr(s) for s in self.symmetries]})>"


class InverseSymmetry(Symmetry):
    """Inverse of a symmetry operation"""

    def __init__(self, delegate):
        super().__init__()
        self.delegate = delegate

    def forward(self, x):
        return self.delegate.inverse(x)

    def inverse(self, x):
        return self.delegate.forward(x)

    def __repr__(self):
        return f"<Inverse: {repr(self.delegate)}>"


class Reflection(Symmetry):
    """Reflection along one dimension."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = x.copy()
        y[..., self.dim] *= -1
        return y

    def inverse(self, x):
        y = x.copy()
        y[..., self.dim] *= -1
        return y

    def __repr__(self):
        return f"<Reflection: {self.dim}>"


class Rotation90(Symmetry):
    """Counterclockwise rotation by 90 degrees."""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        self.mat = np.array(
            [
                [np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                [np.sin(np.pi / 2), np.cos(np.pi / 2)]
            ], dtype=int
        )

    def forward(self, x):
        y = x.copy()
        y[..., [self.dims[0], self.dims[1]]] = np.einsum(
            "ij,...i->...j",
            self.mat,
            y[..., [self.dims[0], self.dims[1]]]
        )
        return y

    def inverse(self, x):
        y = x.copy()
        y[..., [self.dims[0], self.dims[1]]] = np.einsum(
            "ji,...i->...j",
            self.mat,
            y[..., [self.dims[0], self.dims[1]]]
        )
        return y

    def __repr__(self):
        return f"<Rotation90: {self.dims}>"


class Identity(Symmetry):
    """The identity."""

    def forward(self, x):
        return x

    def inverse(self, x):
        return x

    def __repr__(self):
        return "<Identity>"


class SymmetryGroup(list):
    """
    Lattice symmetry group.
    """

    def __init__(self, stencil):
        super().__init__()
        self.stencil = stencil
        candidates = self._make_candidates(stencil.D())
        new_symmetries = {Identity()}
        while len(new_symmetries) > 0:
            for n in new_symmetries:
                if n not in self:
                    self.append(n)
            new_symmetries = self._new_symmetries(candidates)

    @property
    def permutations(self):
        return np.stack([symmetry.permutation(self.stencil) for symmetry in self])

    @property
    def inverse_permutations(self):
        return np.stack([InverseSymmetry(symmetry).permutation(self.stencil) for symmetry in self])

    def moment_representations(self, moment_transform):
        return (moment_transform.matrix[:, self.permutations] @ moment_transform.inverse).swapaxes(0, 1)

    def inverse_moment_representations(self, moment_transform):
        return (moment_transform.matrix[:, self.inverse_permutations] @ moment_transform.inverse).swapaxes(0, 1)

    def _new_symmetries(self, candidates):
        result = []
        for c in candidates:
            for s in self:
                proposed = self._chain_symmetries(s, c)
                if proposed not in self:
                    result.append(proposed)
        return result

    def __contains__(self, symmetry):
        for elem in self:
            if are_symmetries_equal(symmetry, elem, self.stencil):
                return True
        return False

    def index(self, symmetry):
        for i, elem in enumerate(self):
            if are_symmetries_equal(symmetry, elem, self.stencil):
                return i
        raise KeyError("symmetry not in symmetry group")

    def __eq__(self, other):
        assert super.__eq__(other) and self.stencil == other.stencil

    @staticmethod
    def _make_candidates(dim):
        candidates = []
        for i in range(dim):
            for j in range(i + 1, dim):
                candidates.append(Rotation90(i, j))
        for i in range(dim):
            candidates.append(Reflection(i))
        # if for some reason we cannot reach all elements by 90-degree rotations
        for i in range(dim):
            for j in range(i + 1, dim):
                # 180 degree rotations
                candidates.append(ChainedSymmetry(Rotation90(i, j), Rotation90(i, j)))
                # inverse rotations
                candidates.append(InverseSymmetry(Rotation90(i, j)))
        return candidates

    def _chain_symmetries(self, *symmetries):
        symmetries = [
            s for s in symmetries
            if not are_symmetries_equal(s, Identity(), self.stencil)
        ]
        if len(symmetries) == 0:
            return Identity()
        elif len(symmetries) == 1:
            return symmetries[0]
        else:
            return ChainedSymmetry(*symmetries)
