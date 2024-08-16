import pytest
import torch.cuda

from lettuce import Context, Simulation
from lettuce.ext import NoCollision
from tests.common import DummyFlow


class DummyStreaming(DummyFlow):

    def initialize(self):
        self.f.zero_()
        self.f[0, 1, 1] = 1.0
        self.f[1, 2, 2] = 2.0
        self.f[2, 3, 3] = 3.0
        self.f[3, 4, 4] = 4.0
        self.f[4, 5, 5] = 5.0
        self.f[5, 6, 6] = 6.0
        self.f[6, 7, 7] = 7.0
        self.f[7, 8, 8] = 8.0
        self.f[8, 9, 9] = 9.0


def test_native_streaming():
    if not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available on this machine.")
    cpu_context = Context(torch.device('cpu'), use_native=False)
    cpu_flow = DummyStreaming(cpu_context)

    assert cpu_flow.f.shape[0] == 9
    assert cpu_flow.f.shape[1] == 16
    assert cpu_flow.f.shape[2] == 16

    native_context = Context(torch.device('cuda'), use_native=True)
    native_flow = DummyStreaming(native_context)

    assert native_flow.f.shape[0] == 9
    assert native_flow.f.shape[1] == 16
    assert native_flow.f.shape[2] == 16

    collision = NoCollision()

    cpu_simulation = Simulation(cpu_flow, collision, [])
    native_simulation = Simulation(native_flow, collision, [])

    assert cpu_flow.f.cpu().numpy() == pytest.approx(
        native_flow.f.cpu().numpy())

    cpu_simulation(1)
    native_simulation(1)

    assert cpu_flow.f.cpu().numpy() == pytest.approx(
        native_flow.f.cpu().numpy())

    # for i in range(9):
    #     print()
    #     cpu_index = int((cpu_flow.f.cpu()[i, :, :] == float(i + 1)).nonzero(
    #     as_tuple=True)[0])
    #     native_index = int((native_flow.f.cpu()[i, :, :] ==
    #     float(i + 1)).nonzero(as_tuple=True)[0])
    #     print(f"cpu    distribution {i} row {cpu_index}: ", cpu_flow.f.cpu(
    #     )[i, :, :][cpu_index])
    #     print(f"cuda_native distribution {i} row {native_index}: ",
    #     native_flow.f.cpu()[i, :, :][native_index])
