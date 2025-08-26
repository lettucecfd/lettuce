import torch

from ... import Flow, Equilibrium

__all__ = ['IncompressibleQuadraticEquilibrium']


class IncompressibleQuadraticEquilibrium(Equilibrium):
    def __init__(self, rho0=1.0):
        self.rho0 = rho0

    def __call__(self, flow: 'Flow', rho=None, u=None):
        rho = flow.rho() if rho is None else rho
        u = flow.u() if u is None else u

        exu = torch.einsum("qd,d...->q...", [flow.torch_stencil.e, u])
        uxu = torch.einsum("d...,d...->...", [u, u])
        feq = torch.einsum(
             "q...,q...->q...",
             [flow.torch_stencil.w, rho
              + self.rho0 * (
                      (2 * exu - uxu) / (2 * flow.torch_stencil.cs ** 2)
                      + 0.5 * (exu / (flow.torch_stencil.cs ** 2)) ** 2
              )]
        )
        return feq
