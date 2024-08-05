import pytest
import torch.cuda

from lettuce import Context, Simulation
from lettuce.ext import BGKCollision
from tests.common import DummyFlow


class DummyBGK(DummyFlow):

    def initialize(self):
        self.f[:, :, :] = 1.0
        self.f[:, 2, 2] = 2.0


def test_native_bgk_collision():
    if not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available on this machine.")
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

    collision = BGKCollision(2.0)

    cpu_simulation = Simulation(cpu_flow, collision, [])
    native_simulation = Simulation(native_flow, collision, [])

    assert cpu_flow.f.cpu().numpy() == pytest.approx(
        native_flow.f.cpu().numpy())

    cpu_simulation(1)
    native_simulation(1)

    # for i in range(9):
    #     for j in range(16):
    #         print()
    #         print(f"[{i}, {j}, :] cpu ", cpu_flow.f.cpu().numpy()[i, j, :])
    #         print(f"[{i}, {j}, :] nat ", native_flow.f.cpu().numpy()[i, j, :]
    #         )

    for i in range(9):
        for j in range(16):
            assert cpu_flow.f.cpu().numpy()[i, j, :] == pytest.approx(
                native_flow.f.cpu().numpy()[i, j, :]), f"[{i}, {j}, :]"

    #     print()
    #     cpu_index = int((cpu_flow.f.cpu()[i, :, :] == float(i + 1)).
    #     nonzero(as_tuple=True)[0])
    #     native_index = int((native_flow.f.cpu()[i, :, :] == float(i + 1)).
    #     nonzero(as_tuple=True)[0])
    #     print(f"cpu    distribution {i} row {cpu_index}: ", cpu_flow.f.cpu()
    #     [i, :, :][cpu_index])
    #     print(f"cuda_native distribution {i} row {native_index}: ",
    #     native_flow.f.cpu()[i, :, :][native_index])
