import pytest
import torch.cuda

from lettuce import Context, Simulation, StreamingStrategy
from lettuce.ext import BGKCollision
from .test_native_bgk_collision import DummyBGK


@pytest.mark.parametrize("streaming_strategy",
                         [StreamingStrategy.NO_STREAMING, StreamingStrategy.PRE_STREAMING, StreamingStrategy.POST_STREAMING, StreamingStrategy.DOUBLE_STREAMING])
def test_native_streaming_strategy(streaming_strategy):
    cpu_context = Context('cpu')
    cpu_flow = DummyBGK(cpu_context)

    assert cpu_flow.f.shape[0] == 9
    assert cpu_flow.f.shape[1] == 16
    assert cpu_flow.f.shape[2] == 16

    native_context = Context(use_native=True)
    native_flow = DummyBGK(native_context)

    assert native_flow.f.shape[0] == 9
    assert native_flow.f.shape[1] == 16
    assert native_flow.f.shape[2] == 16

    collision = BGKCollision(2.0)

    cpu_simulation = Simulation(cpu_flow, collision, [], streaming_strategy)
    native_simulation = Simulation(native_flow, collision, [], streaming_strategy)

    assert cpu_flow.f.cpu().numpy() == pytest.approx(native_flow.f.cpu().numpy())

    cpu_simulation(1)
    native_simulation(1)

    for i in range(9):
        for j in range(16):
            assert cpu_flow.f.cpu().numpy()[i, j, :] == pytest.approx(
                native_flow.f.cpu().numpy()[i, j, :]), f"[{i}, {j}, :]"
