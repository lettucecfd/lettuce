import torch

from ... import Flow, Equilibrium

__all__ = ['QuadraticEquilibrium']

from ...native.ext import NativeQuadraticEquilibrium


class QuadraticEquilibrium(Equilibrium):
    def __call__(self, flow: 'Flow', rho=None, u=None):
        rho = flow.rho() if rho is None else rho
        u = flow.u() if u is None else u

        exu = torch.tensordot(flow.torch_stencil.e, u, dims=1)
        uxu = flow.einsum("d,d->", [u, u])
        feq = flow.einsum("q,q->q",
                          [flow.torch_stencil.w, rho * ((2 * exu - uxu) / (2 * flow.torch_stencil.cs ** 2) + 0.5 * (exu / (flow.torch_stencil.cs ** 2)) ** 2 + 1)])

        return feq

    def native_available(self) -> bool:
        return True

    def native_generator(self) -> 'NativeEquilibrium':
        return NativeQuadraticEquilibrium()
