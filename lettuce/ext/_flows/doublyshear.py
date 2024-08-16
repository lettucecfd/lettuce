"""
Doubly shear layer in 2D.
Special Inputs & standard value: shear_layer_width = 80,
initial_perturbation_magnitude = 0.05
"""
from typing import Union, List, Optional

import numpy as np
import torch

from lettuce._unit import UnitConversion
from . import ExtFlow

__all__ = ['DoublyPeriodicShear2D']

from .._stencil import D2Q9


class DoublyPeriodicShear2D(ExtFlow):

    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 shear_layer_width=80,
                 initial_perturbation_magnitude=0.05):
        self.initial_perturbation_magnitude = initial_perturbation_magnitude
        self.shear_layer_width = shear_layer_width
        self.stencil = D2Q9() if stencil is None else stencil
        super().__init__(context, resolution, reynolds_number, mach_number,
                         self.stencil, equilibrium)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * self.stencil.d
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

    def analytic_solution(self, x, t=0):
        raise NotImplementedError

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        pert = self.initial_perturbation_magnitude
        w = self.shear_layer_width
        u1 = self.context.convert_to_tensor(torch.where(
            self.grid[1] > 0.5,
            torch.tanh(w * (self.grid[1] - 0.25)),
            torch.tanh(w * (0.75 - self.grid[1]))
        ))
        u2 = pert * torch.sin(2 * torch.pi * (self.grid[0] + 0.25))
        u = torch.stack([u1, u2])
        p = torch.zeros_like(u1)[None, ...]
        return p, u

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
        return []
