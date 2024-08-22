"""
Test boundary conditions.
"""

from lettuce import *
from lettuce.ext import *
from .common import DummyTGV

import pytest

import numpy as np
import torch


class my_equilibrium_boundary_mask(EquilibriumBoundaryPU):

    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        a = context.one_tensor(shape, dtype=bool)
        return a

    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        return context.one_tensor(shape, dtype=bool)


class DummyEQBC(TaylorGreenVortex):
    @property
    def boundaries(self):
        u = self.context.one_tensor([2, 1, 1]) * 0.1
        p = self.context.zero_tensor([1, 1, 1])
        boundary = my_equilibrium_boundary_mask(self.context,
                                                torch.ones(self.resolution),
                                                u, p)
        return [boundary]


def test_equilibrium_boundary_pu():
    context = Context(device=torch.device('cpu'), dtype=torch.float64,
                      use_native=False)

    flow_1 = DummyEQBC(context, resolution=[16, 16], reynolds_number=1,
                       mach_number=0.1)
    flow_2 = DummyTGV(context, resolution=[16, 16], reynolds_number=1,
                      mach_number=0.1)

    simulation = Simulation(flow=flow_1, collision=NoCollision(), reporter=[])
    simulation(num_steps=1)

    pressure = 0
    velocity = 0.1 * np.ones(flow_2.stencil.d)

    feq = flow_2.equilibrium(
        flow_2,
        context.convert_to_tensor(
            flow_2.units.convert_pressure_pu_to_density_lu(pressure)),
        context.convert_to_tensor(
            flow_2.units.convert_velocity_to_lu(velocity))
    )
    flow_2.f = torch.einsum("q,q...->q...", feq, torch.ones_like(flow_2.f))

    assert flow_1.f.cpu().numpy() == pytest.approx(flow_2.f.cpu().numpy())
