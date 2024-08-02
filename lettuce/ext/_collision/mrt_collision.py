from .. import Force
from ... import Flow, Collision

__all__ = ['MRTCollision']


class MRTCollision(Collision):
    """
    Multiple relaxation time _collision operator

    This is an MRT operator in the most general sense of the word.
    The transform does not have to be linear and can, e.g., be any moment or cumulant transform.
    """

    def __init__(self, lattice, transform, relaxation_parameters):
        Collision.__init__(self, lattice)
        self.lattice = lattice
        self.transform = transform
        self.relaxation_parameters = lattice.convert_to_tensor(relaxation_parameters)

    def __call__(self, f):
        m = self.transform.transform(f)
        meq = self.transform.equilibrium(m)
        m = m - self.lattice.einsum("q,q->q", [1 / self.relaxation_parameters, m - meq])
        f = self.transform.inverse_transform(m)
        return f
