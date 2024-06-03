from lettuce import *
from lettuce.ext import *

import pytest

import numpy as np
import torch

from tests.conftest import TestFlow


def input_moment_dimensions_generator_(d):
    if d == 1:
        yield [1]
        yield [16]
    else:
        for x in input_moment_dimensions_generator_(d - 1):
            yield x + [1]
            yield x + [16]


def input_moment_dimensions_generator():
    for d in range(1, 4):
        for x in input_moment_dimensions_generator_(d):
            yield x


def input_moment_dimension_ids():
    return ['x'.join([str(k) for k in it]) for it in input_moment_dimensions_generator()]


@pytest.fixture(params=input_moment_dimensions_generator(),
                ids=input_moment_dimension_ids())
def input_moment_dimensions(request):
    return request.param


class TestEquilibriumBoundary(EquilibriumBoundaryPU):

    def make_no_collision_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        m = context.zero_tensor(shape, dtype=bool)
        m[..., :1] = True
        return m

    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        return context.one_tensor(shape, dtype=bool)


def test_equilibrium_boundary_pu_algorithm(stencils, configurations):
    """
    Test for the equilibrium boundary algorithm. This test verifies that the algorithm correctly computes the
    equilibrium outlet pressure by comparing its output to manually calculated equilibrium values.
    """

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
    flow_2.f[:, :1, ...] = torch.einsum("q,q...->q...", feq, torch.ones_like(flow_2.f))[:, :8, ...]

    assert flow_1.f.cpu().numpy() == pytest.approx(flow_2.f.cpu().numpy())


def test_equilibrium_boundary_pu_native(input_moment_dimensions):
    context_native = Context(device=torch.device('cuda'), dtype=torch.float64, use_native=True)
    context_cpu = Context(device=torch.device('cpu'), dtype=torch.float64, use_native=False)

    flow_native = TestFlow(context_native, resolution=[16] * len(input_moment_dimensions), reynolds_number=1, mach_number=0.1)
    flow_cpu = TestFlow(context_cpu, resolution=[16] * len(input_moment_dimensions), reynolds_number=1, mach_number=0.1)

    u = np.ones([2] + input_moment_dimensions)
    rho = np.ones([1] + input_moment_dimensions)

    boundary_native = TestEquilibriumBoundary(context_native, u, rho)
    boundary_cpu = TestEquilibriumBoundary(context_cpu, u, rho)

    simulation_native = Simulation(flow=flow_native, collision=NoCollision(), boundaries=[boundary_native], reporter=[])
    simulation_cpu = Simulation(flow=flow_cpu, collision=NoCollision(), boundaries=[boundary_cpu], reporter=[])

    simulation_native(num_steps=1)
    simulation_cpu(num_steps=1)

    assert flow_cpu.f.cpu().numpy() == pytest.approx(flow_native.f.cpu().numpy(), rel=1e-3)
