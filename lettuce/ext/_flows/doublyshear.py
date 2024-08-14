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


class DoublyPeriodicShear2D(ExtFlow):

    def __init__(self, context: 'Context', resolution, reynolds_number,
                 mach_number, shear_layer_width=80,
                 initial_perturbation_magnitude=0.05):
        super().__init__(context, resolution, reynolds_number, mach_number)
        self.initial_perturbation_magnitude = initial_perturbation_magnitude
        self.shear_layer_width = shear_layer_width

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
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        raise NotImplementedError

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        pert = self.initial_perturbation_magnitude
        w = self.shear_layer_width
        u1 = self.context.convert_to_tensor(np.choose(
            self.grid[1] > 0.5,
            [np.tanh(w * (self.grid[1] - 0.25)),
             np.tanh(w * (0.75 - self.grid[1]))]
        ))
        u2 = self.context.convert_to_tensor(
            pert * np.sin(2 * np.pi * (self.grid[0] + 0.25)))
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
