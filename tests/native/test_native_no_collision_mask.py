from tests.common import *

def test_native_no_collision_mask():
    """test if """

    if not torch.cuda.is_available():
        pytest.skip("Native test skipped")

    context = Context(dtype=torch.float32, device=torch.device("cuda"),
                      use_native=True)
    flow = TestFlow(context, 16, 1, 0.01)

    collision = BGKCollision(1.0)
    simulation = Simulation(flow, collision, [])
    simulation.no_streaming_mask = context.one_tensor(flow.resolution,
                                                      dtype=bool)

    f0 = copy(flow.f)
    simulation(64)
    f1 = flow.f

    assert torch.isclose(f0, f1).all()