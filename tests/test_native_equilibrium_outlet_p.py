"""
Test boundary conditions.
"""

from lettuce import *
from lettuce.ext import *

import pytest

import numpy as np
import torch

# TODO: Implement native generator and test suite

class TestFlow(ExtFlow):

    def make_resolution(self, resolution: Union[int, List[int]]) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * 2
        else:
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(reynolds_number, mach_number, characteristic_length_lu=resolution[0])

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        u = np.ones([2] + self.resolution) * 1
        p = np.ones([1] + self.resolution) * 1
        return p, u

def test_equilibrium_outlet_p():
    context = Context(device=torch.device('cpu'), dtype=torch.float64, use_native=False)
    flow = TestFlow(context, resolution=4, reynolds_number=1, mach_number=0.1)
    boundary_cpu = EquilibriumOutletP(flow=flow, context=context, direction=[1, 0], rho_outlet=1.2)
    f_post_boundary = boundary_cpu(flow)[:, -1, :]
    u = flow.units.convert_velocity_to_lu(context.one_tensor([2, 1, flow.resolution[1]]))
    rho = context.one_tensor([1, 1, 1]) * 1.2
    reference = flow.equilibrium(flow, rho=rho, u=u)[:, 0, :]
    assert reference.cpu().numpy() == pytest.approx(f_post_boundary.cpu().numpy(), rel=1e-6)
