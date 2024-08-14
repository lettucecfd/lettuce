from ... import Flow, Collision

__all__ = ['MRTCollision']


class MRTCollision(Collision):
    """
    Multiple relaxation time collision operator

    This is an MRT operator in the most general sense of the word.
    The transform does not have to be linear and can, e.g., be any moment or
    cumulant transform.
    """

    def __init__(self, transform: 'Transform', relaxation_parameters: list,
                 context: 'Context'):
        self.transform = transform
        self.relaxation_parameters = context.convert_to_tensor(
            relaxation_parameters)

    def __call__(self, flow: 'Flow'):
        m = self.transform.transform(flow.f)
        meq = self.transform.equilibrium(m)
        m = m - flow.einsum("q,q->q", [1 / self.relaxation_parameters,
                                       m - meq])
        f = self.transform.inverse_transform(m)
        return f

    def native_available(self) -> bool:
        return False

    def native_generator(self) -> 'NativeCollision':
        pass
