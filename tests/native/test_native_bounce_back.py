from typing import List, Optional

import pytest
import torch.cuda

from lettuce import Context, Simulation
from lettuce.ext import NoCollision, BounceBackBoundary
from tests.conftest import DummyFlow


class MyBounceBackBoundary(BounceBackBoundary):
    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        mask = context.zero_tensor(shape, dtype=bool)
        mask[0, :] = True
        mask[:, 0] = True
        mask[2:, :] = True
        mask[:, 2:] = True
        return mask


class DummyBBBC(DummyFlow):

    def initialize(self):
        self.f.zero_()
        self.f[:, 1, 1] = 1.0

    @property
    def boundaries(self) -> List['Boundary']:
        return [MyBounceBackBoundary(torch.ones(self.resolution))]


def test_native_bounce_back():
    cpu_context = Context(torch.device('cpu'), use_native=False)
    cpu_flow = DummyBBBC(cpu_context)

    assert cpu_flow.f.shape[0] == 9
    assert cpu_flow.f.shape[1] == 16
    assert cpu_flow.f.shape[2] == 16

    native_context = Context(torch.device('cuda'), use_native=True)
    native_flow = DummyBBBC(native_context)

    assert native_flow.f.shape[0] == 9
    assert native_flow.f.shape[1] == 16
    assert native_flow.f.shape[2] == 16

    collision = NoCollision()

    cpu_simulation = Simulation(cpu_flow, collision, [])
    native_simulation = Simulation(native_flow, collision, [])

    assert cpu_flow.f.cpu().numpy() == pytest.approx(
        native_flow.f.cpu().numpy())

    # print()
    # for i in range(9):
    #     print(native_flow.f.cpu()[i, 0:4, 0:4])

    cpu_simulation(1)
    native_simulation(1)

    assert cpu_flow.f.cpu().numpy() == pytest.approx(
        native_flow.f.cpu().numpy())

    # print()
    # for i in range(9):
    #     print(native_flow.f.cpu()[i, 0:4, 0:4])

    cpu_simulation(1)
    native_simulation(1)

    assert cpu_flow.f.cpu().numpy() == pytest.approx(
        native_flow.f.cpu().numpy())

    # print()
    # for i in range(9):
    #     print(native_flow.f.cpu()[i, 0:4, 0:4])
