"""
Test boundary conditions.
"""

from lettuce import *
from lettuce.ext import *

import pytest

import numpy as np
import torch


class my_equilibrium_boundary_mask(EquilibriumBoundaryPU):

    def make_no_collision_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        a = context.one_tensor(shape, dtype=bool)
        # a[:4,:4] = True
        return a
        # return None

    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        return context.one_tensor(shape, dtype=bool)
        # return None


class my_basic_flow(ExtFlow):

    def make_resolution(self, resolution: Union[int, List[int]], stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * 2
        else:
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(reynolds_number, mach_number, characteristic_length_lu=resolution[0])

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution[0], endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution[1], endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        grid = self.grid
        t = 0
        nu = self.units.viscosity_pu
        u = np.array([np.cos(grid[0]) * np.sin(grid[1]) * np.exp(-2 * nu * t),
                      -np.sin(grid[0]) * np.cos(grid[1]) * np.exp(-2 * nu * t)]) * 0
        p = -np.array([0.25 * (np.cos(2 * grid[0]) + np.cos(2 * grid[1])) * np.exp(-4 * nu * t)]) * 0
        return p, u


def test_equilibrium_boundary_pu():
    context = Context(device=torch.device('cpu'), dtype=torch.float64, use_native=False)

    flow_1 = my_basic_flow(context, resolution=16, reynolds_number=1, mach_number=0.1)
    flow_2 = my_basic_flow(context, resolution=16, reynolds_number=1, mach_number=0.1)

    boundary = my_equilibrium_boundary_mask(context, [0.1, 0.1], 0)

    u = context.one_tensor([2, 1, 1]) * 0.1
    p = context.zero_tensor([1, 1, 1])
    boundary = my_equilibrium_boundary_mask(context, u, p)

    simulation = Simulation(flow=flow_1, collision=NoCollision(), boundaries=[boundary], reporter=[])
    simulation(num_steps=1)

    pressure = 0
    velocity = 0.1 * np.ones(flow_2.stencil.d)
    # stencil = D2Q9()
    # u_slice = [stencil.d, *flow_2.resolution[:stencil.d-1],1]
    # p_slice = [1,*flow_2.resolution[:stencil.d-1],1]
    # u = flow_2.units.convert_velocity_to_lu(context.one_tensor(u_slice))
    # p = context.one_tensor(p_slice) * 1.2

    feq = flow_2.equilibrium(
        flow_2,
        context.convert_to_tensor(flow_2.units.convert_pressure_pu_to_density_lu(pressure)),
        context.convert_to_tensor(flow_2.units.convert_velocity_to_lu(velocity))
    )
    flow_2.f = torch.einsum("q,q...->q...", feq, torch.ones_like(flow_2.f))

    assert flow_1.f.cpu().numpy() == pytest.approx(flow_2.f.cpu().numpy())
