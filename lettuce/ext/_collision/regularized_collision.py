import torch

from .. import Force
from ... import Flow, Collision

__all__ = ['RegularizedCollision']


class RegularizedCollision(Collision):
    """
    Regularized LBM according to Jonas Latt and Bastien Chopard (2006)
    """

    def __init__(self, tau):
        self.tau = tau
        self.Q_matrix = None

    def __call__(self, flow: 'Flow'):
        if self.Q_matrix is None:
            self.Q_matrix = torch.zeros([flow.stencil.q, flow.stencil.d,
                                         flow.stencil.d],
                                        device=flow.context.device,
                                        dtype=flow.context.dtype)

            for a in range(flow.stencil.q):
                for b in range(flow.stencil.d):
                    for c in range(flow.stencil.d):
                        self.Q_matrix[a, b, c] = (
                                flow.torch_stencil.e[a, b]
                                * flow.torch_stencil.e[a, c])
                        if b == c:
                            self.Q_matrix[a, b, c] -= (flow.torch_stencil.cs
                                                       ** 2)
        feq = flow.equilibrium(flow)
        pi_neq = flow.shear_tensor(flow.f - feq)
        cs4 = flow.stencil.cs ** 4

        pi_neq = flow.einsum("qab,ab->q", [self.Q_matrix, pi_neq])
        pi_neq = flow.einsum("q,q->q", [flow.torch_stencil.w, pi_neq])

        fi1 = pi_neq / (2 * cs4)
        f = feq + (1. - 1. / self.tau) * fi1

        return f
