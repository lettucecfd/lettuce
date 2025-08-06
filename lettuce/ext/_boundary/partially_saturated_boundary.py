from typing import List
import torch
from ... import Boundary


class PartiallySaturatedBC(Boundary):
    """
    Partially saturated boundary condition using a partial combination of
    standard full-way bounce back and BGK-Collision, first presented by Noble
    and Torczynski (1998), see Krüger et al., pp. 448.
    This may be just as efficient as a compact version, because the boundary is
    actually used on all nodes even within the object.
    """

    def __init__(self, mask: torch.Tensor, tau: float, saturation: float):
        self._mask = mask
        self.tau = tau
        # B(epsilon, theta), Krüger p. 448ff
        self.B = saturation * (tau - 0.5) / ((1 - saturation) + (tau - 0.5))
        return

    def __call__(self, flow: 'Flow'):
        feq = flow.equilibrium(flow)
        # TODO: benchmark and possibly use indices (like _compact)
        # and/or calculate feq twice within torch.where (like _less_memory)
        f = torch.where(
            self._mask,
            flow.f - (1.0 - self.B) / self.tau * (flow.f - feq)
            + self.B * ((flow.f[flow.stencil.opposite]
                         - feq[flow.stencil.opposite])
                        - (flow.f
                           - flow.equilibrium(flow,
                                              u=torch.zeros_like(flow.u()))
                           )
                        ),
            flow.f)
        return f

    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> torch.Tensor:
        return self._mask

    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ):
        pass

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int) -> 'NativeBoundary':
        pass
