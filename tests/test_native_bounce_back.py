from typing import List, Union, Optional

import pytest
import torch.cuda

from lettuce import Context, UnitConversion, Simulation
from lettuce.ext import ExtFlow, NoCollision, BounceBackBoundary


class MyBounceBackBoundary(BounceBackBoundary):
    def make_no_collision_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        mask = context.zero_tensor(shape, dtype=bool)
        mask[0, :] = True
        mask[:, 0] = True
        mask[2:, :] = True
        mask[:, 2:] = True
        return mask


class DummyFlow(ExtFlow):
    def __init__(self, context: Context):
        ExtFlow.__init__(self, context, 16, 1.0, 1.0)

    def make_resolution(self, resolution: Union[int, List[int]]) -> List[int]:
        return [resolution, resolution] if isinstance(resolution, int) else resolution

    def make_units(self, reynolds_number, mach_number, _: List[int]) -> 'UnitConversion':
        return UnitConversion(reynolds_number=reynolds_number, mach_number=mach_number)

    def initial_pu(self) -> (float, List[float]):
        ...

    def initialize(self):
        self.f.zero_()
        self.f[:, 1, 1] = 1.0


def test_native_bounce_back():
    cpu_context = Context(torch.device('cpu'), use_native=False)
    cpu_flow = DummyFlow(cpu_context)

    assert cpu_flow.f.shape[0] == 9
    assert cpu_flow.f.shape[1] == 16
    assert cpu_flow.f.shape[2] == 16

    native_context = Context(torch.device('cuda'), use_native=True)
    native_flow = DummyFlow(native_context)

    assert native_flow.f.shape[0] == 9
    assert native_flow.f.shape[1] == 16
    assert native_flow.f.shape[2] == 16

    collision = NoCollision()
    boundaries = [MyBounceBackBoundary()]

    cpu_simulation = Simulation(cpu_flow, collision, boundaries, [])
    native_simulation = Simulation(native_flow, collision, boundaries, [])

    assert cpu_flow.f.cpu().numpy() == pytest.approx(native_flow.f.cpu().numpy())

    # print()
    # for i in range(9):
    #     print(native_flow.f.cpu()[i, 0:4, 0:4])

    cpu_simulation(1)
    native_simulation(1)

    assert cpu_flow.f.cpu().numpy() == pytest.approx(native_flow.f.cpu().numpy())

    # print()
    # for i in range(9):
    #     print(native_flow.f.cpu()[i, 0:4, 0:4])

    cpu_simulation(1)
    native_simulation(1)

    assert cpu_flow.f.cpu().numpy() == pytest.approx(native_flow.f.cpu().numpy())

    # print()
    # for i in range(9):
    #     print(native_flow.f.cpu()[i, 0:4, 0:4])
