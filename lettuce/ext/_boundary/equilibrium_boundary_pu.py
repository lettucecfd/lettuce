import numbers
from typing import List, Optional, Union

import numpy as np
import torch

from ... import Boundary
from ... import Context
from ... import Flow
from ...cuda_native.ext import NativeEquilibriumBoundaryPu

__all__ = ['EquilibriumBoundaryPU']


class EquilibriumBoundaryPU(Boundary):
    """Sets distributions on this boundary to equilibrium with predefined
    velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes
    equations. This boundary condition should only be used if no better
    options are available.
    """

    @staticmethod
    def checked_tensor(t: Union[torch.Tensor, np.ndarray, list, tuple, numbers.Number],
                       context: 'Context', flow: 'Flow') -> torch.Tensor:

        # 1) Base conversion to a tensor
        if not torch.is_tensor(t):
            if isinstance(t, numbers.Number):
                t = torch.tensor(t)  # scalar → zero-dim tensor
            elif isinstance(t, (np.ndarray, list, tuple)):
                t = torch.as_tensor(t)  # array-like → tensor
            else:
                raise TypeError(f"Cannot convert {type(t)} to tensor")
        # now t is a tensor of some dtype; we’ll set dtype/device at the end

        d = flow.stencil.d
        spatial = list(flow.f.shape[1:])  # e.g. [16,16]
        full_ndim = 1 + len(spatial)

        # 2) Expand scalars → [1,1,...,1]
        if t.ndim == 0:
            t = t.reshape([1] * full_ndim)

        # 3) Expand a length-d vector → [d,1,1,...,1]
        elif t.ndim == 1 and t.shape[0] == d:
            t = t.reshape([d] + [1] * len(spatial))

        # 4) Expand a pure spatial field → [1, N₁, N₂, ...]
        elif t.ndim == len(spatial) and list(t.shape) == spatial:
            t = t.unsqueeze(0)  # prepend a 1 in the component axis

        # 5) If it’s already [d, N₁,...,Nₖ], accept it; otherwise error
        elif t.ndim != full_ndim:
            raise ValueError(f"Tensor has wrong rank: expected {full_ndim}, got {t.ndim}")

        # 6) Check broadcast compatibility
        #    dim 0 must be d or 1; dims 1... must be N_i or 1
        for i in range(full_ndim):
            size = t.shape[i]
            if i == 0:
                if size not in (1, d):
                    raise ValueError(f"Component dim must be 1 or {d}, got {size}")
            else:
                if size not in (1, spatial[i - 1]):
                    raise ValueError(f"Spatial dim {i} must be 1 or {spatial[i - 1]}, got {size}")

        # 7) Finally send through context.convert_to_tensor to pick up dtype/device
        return context.convert_to_tensor(t)

    def __init__(self, context: 'Context', flow: 'Flow', mask, velocity, pressure=0):
        self.velocity = self.checked_tensor(velocity, context, flow)
        self.pressure = self.checked_tensor(pressure, context, flow)
        # velocity = [velocity] if not hasattr(velocity, 'len') else velocity
        # self.velocity = context.convert_to_tensor(velocity)
        # self.pressure = context.convert_to_tensor(pressure)
        self._mask = mask

    def __call__(self, flow: 'Flow'):
        rho = flow.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = flow.units.convert_velocity_to_lu(self.velocity)
        feq = flow.equilibrium(flow, rho, u)
        feq = torch.einsum("q...,q...->q...", [feq, torch.ones_like(flow.f)])
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
