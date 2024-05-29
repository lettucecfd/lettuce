from lettuce import *
from lettuce.ext import *

import pytest

import numpy as np
import torch

from tests.conftest import TestFlow


class TestEquilibriumBoundary(EquilibriumBoundaryPU):

    def make_no_collision_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        a = context.zero_tensor(shape, dtype=bool)
        a[:8, ...] = True
        return a
        # return None

    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        return context.one_tensor(shape, dtype=bool)
        # return None


def test_equilibrium_boundary_pu_algorithm(stencils, configurations):
    '''
    Test for the equilibrium boundary algorithm. This test verifies that the algorithm correctly computes the
    equilibrium outlet pressure by comparing its output to manually calculated equilibrium values.
    '''
    dtype, device, native = configurations
    context = Context(device=torch.device(device), dtype=dtype, use_native=(native == "native"))

    stencil = stencils()
    flow_1 = TestFlow(context, resolution=stencil.d * [16], reynolds_number=1, mach_number=0.1, stencil=stencil)
    flow_2 = TestFlow(context, resolution=stencil.d * [16], reynolds_number=1, mach_number=0.1, stencil=stencil)

    u_slice = [stencil.d, *flow_2.resolution[:stencil.d - 1], 1]
    p_slice = [1, *flow_2.resolution[:stencil.d - 1], 1]
    u = context.one_tensor(u_slice) * 0.1
    p = context.zero_tensor(p_slice)

    boundary = TestEquilibriumBoundary(context, u, p)
    simulation = Simulation(flow=flow_1, collision=NoCollision(), boundaries=[boundary], reporter=[])
    simulation(num_steps=1)

    pressure = 0
    velocity = 0.1 * np.ones(flow_2.stencil.d)

    feq = flow_2.equilibrium(
        flow_2,
        context.convert_to_tensor(flow_2.units.convert_pressure_pu_to_density_lu(pressure)),
        context.convert_to_tensor(flow_2.units.convert_velocity_to_lu(velocity))
    )
    flow_2.f[:, :8, ...] = torch.einsum("q,q...->q...", feq, torch.ones_like(flow_2.f))[:, :8, ...]

    assert flow_1.f.cpu().numpy() == pytest.approx(flow_2.f.cpu().numpy())


def test_equilibrium_boundary_pu_native():
    context_native = Context(device=torch.device('cuda'), dtype=torch.float64, use_native=True)
    context_cpu = Context(device=torch.device('cpu'), dtype=torch.float64, use_native=False)

    stencil = D2Q9()
    flow_native = TestFlow(context_native, resolution=stencil.d * [16], reynolds_number=1, mach_number=0.1,
                           stencil=stencil)
    flow_cpu = TestFlow(context_cpu, resolution=stencil.d * [16], reynolds_number=1, mach_number=0.1, stencil=stencil)

    '''Works as expected'''
    u = np.ones([2, 16, 16])
    rho = np.ones([16, 16])

    '''Does not work'''
    # u = np.ones([2,16,1])
    # rho = np.ones([16,1])

    '''Does not work'''
    # u = np.ones([2,1,1])
    # rho = np.ones([1,1])

    boundary_native = TestEquilibriumBoundary(context_native, u, rho)
    boundary_cpu = TestEquilibriumBoundary(context_cpu, u, rho)

    simulation_native = Simulation(flow=flow_native, collision=NoCollision(), boundaries=[boundary_native], reporter=[])
    simulation_cpu = Simulation(flow=flow_cpu, collision=NoCollision(), boundaries=[boundary_cpu], reporter=[])

    simulation_native(num_steps=1)
    simulation_cpu(num_steps=1)

    print()
    print("Print the first 4 rows/columns of the velocities ux and uy for better visualization and comparison:")
    print("Native:")
    print(flow_native.velocity[:, :4, :4])
    print("CPU:")
    print(flow_cpu.velocity[:, :4, :4])
    assert flow_cpu.f.cpu().numpy() == pytest.approx(flow_native.f.cpu().numpy(), rel=1e-6)
