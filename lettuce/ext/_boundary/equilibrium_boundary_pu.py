from typing import List, Optional

import torch

from ... import Boundary
from ... import Flow
from ... import Context
from ...cuda_native.ext import NativeEquilibriumBoundaryPu

__all__ = ['EquilibriumBoundaryPU']


class EquilibriumBoundaryPU(Boundary):
    """Sets distributions on this boundary to equilibrium with predefined
    velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes
    equations. This boundary condition should only be used if no better
    options are available.
    """

    def __init__(self, context: 'Context', mask, velocity, pressure=0):
        self.velocity = context.convert_to_tensor(velocity)
        self.pressure = context.convert_to_tensor(pressure)
        self._mask = mask

    def __call__(self, flow: 'Flow'):
        rho = flow.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = flow.units.convert_velocity_to_lu(self.velocity)
        feq = flow.equilibrium(flow, rho, u)
        feq = flow.einsum("q,q->q", [feq, torch.ones_like(flow.f)])
        return feq

    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        return self._mask

    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        pass

    def native_available(self) -> bool:
        return True

    def native_generator(self, index: int) -> 'NativeBoundary':
        return NativeEquilibriumBoundaryPu(index)
