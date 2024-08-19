"""
Cavity flow
"""
from typing import List, Union, Optional

import numpy as np
import torch

from ... import UnitConversion
from .. import BounceBackBoundary, EquilibriumBoundaryPU
from ._ext_flow import ExtFlow


class Cavity2D(ExtFlow):

    def __init__(self, context: 'Context', resolution, reynolds_number,
                 mach_number):
        super().__init__(context, resolution, reynolds_number, mach_number)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * 2
        else:
            assert len(resolution) == 2, 'expected 2-dimensional resolution'
            return resolution

    def make_units(self, reynolds_number, mach_number,
                   resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution[0], characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_pu(self):
        return (torch.stack([torch.zeros_like(self.grid[0])]),
                torch.stack([torch.zeros_like(self.grid[0])] * 2))  # p, u

    @property
    def grid(self) -> (torch.Tensor, torch.Tensor):
        endpoints = [1 - 1 / n for n in
                     self.resolution]  # like endpoint=False in np.linspace
        xyz = tuple(torch.linspace(0, endpoints[n],
                                   steps=self.resolution[n],
                                   device=self.context.device,
                                   dtype=self.context.dtype)
                    for n in range(self.stencil.d))
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        x, *y = self.grid
        boundary = self.context.zero_tensor(x.shape, dtype=bool)
        top = self.context.zero_tensor(x.shape, dtype=bool)
        boundary[[0, -1], 1:] = True  # left and right
        boundary[:, 0] = True  # bottom
        top[:, -1] = True  # top
        return [
            # bounce back walls
            BounceBackBoundary(boundary),
            # moving fluid on top# moving bounce back top
            EquilibriumBoundaryPU(
                self.context, top, 
                [float(self.units.characteristic_velocity_pu), 0.0]
            ),
        ]
