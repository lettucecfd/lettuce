import torch

from ... import Flow, Equilibrium

__all__ = ['QuadraticEquilibriumLessMemory']


class QuadraticEquilibriumLessMemory(Equilibrium):
    """
    does the same as the normal equilibrium, however it uses somewhere around
    20% less RAM, but runs about 2% slower on GPU and 11% on CPU
    """

    def __call__(self, flow: 'Flow', rho=None, u=None):
        rho = flow.rho() if rho is None else rho
        u = flow.u() if u is None else u

        feq = flow.einsum(
            "q,q->q",
            [flow.torch_stencil.w,
             rho * ((2 * torch.tensordot(flow.torch_stencil.e, u, dims=1)
                     - flow.einsum("d,d->", [u, u]))
                    / (2 * flow.torch_stencil.cs ** 2)
                    + 0.5 * (torch.tensordot(flow.torch_stencil.e, u, dims=1)
                             / (flow.torch_stencil.cs ** 2)) ** 2 + 1
                    )
             ]
        )
        return feq

    def native_available(self) -> bool:
        return False

    def native_generator(self) -> 'NativeEquilibrium':
        pass
