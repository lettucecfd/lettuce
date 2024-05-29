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
        a = context.zero_tensor(shape, dtype=bool)
        a[:,1] = True
        return a
        # return None

    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        return context.one_tensor(shape, dtype=bool)
        # return None


class my_basic_flow(ExtFlow):

    def make_resolution(self, resolution: Union[int, List[int]]) -> List[int]:
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
                      -np.sin(grid[0]) * np.cos(grid[1]) * np.exp(-2 * nu * t)])*0+1
        p = -np.array([0.25 * (np.cos(2 * grid[0]) + np.cos(2 * grid[1])) * np.exp(-4 * nu * t)])*0
        return p, u

def test_equilibrium_boundary_pu_native():
    context_native = Context(device=torch.device('cuda'), dtype=torch.float64, use_native=True)
    context_cpu = Context(device=torch.device('cpu'), dtype=torch.float64, use_native=False)

    flow_native = my_basic_flow(context_native, resolution=16, reynolds_number=1, mach_number=0.1)
    flow_cpu = my_basic_flow(context_cpu, resolution=16, reynolds_number=1, mach_number=0.1)

    '''Works as expected'''
    u = np.ones([2,16,16])
    rho = np.ones([16,16])

    '''Does not work'''
    # u = np.ones([2,16,1])
    # rho = np.ones([16,1])

    '''Does not work'''
    # u = np.ones([2,1,1])
    # rho = np.ones([1,1])

    boundary_native = my_equilibrium_boundary_mask(context_native, u, rho)
    boundary_cpu = my_equilibrium_boundary_mask(context_cpu, u, rho)

    simulation_native = Simulation(flow=flow_native, collision=NoCollision(), boundaries=[boundary_native], reporter=[])
    simulation_cpu = Simulation(flow=flow_cpu, collision=NoCollision(), boundaries=[boundary_cpu], reporter=[])

    simulation_native(num_steps=1)
    simulation_cpu(num_steps=1)

    print()
    print("Print the first 4 rows/columns of the velocities ux and uy for better visualization and comparison:")
    print("Native:")
    print(flow_native.velocity[:,:4,:4])
    print("CPU:")
    print(flow_cpu.velocity[:,:4,:4])
    assert flow_cpu.f.cpu().numpy() == pytest.approx(flow_native.f.cpu().numpy(), rel=1e-6)
