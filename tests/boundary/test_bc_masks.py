from tests.common import *

def test_masks(fix_configuration):
    """test if masks are applied from boundary conditions"""
    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)
    flow = Obstacle(context=context, resolution=[16, 16], reynolds_number=100,
                    mach_number=0.1, domain_length_x=2)
    all_native_boundaries_in_Obstacle = sum([
        _.native_available() for _ in flow.boundaries]) == flow.boundaries
    if native and not all_native_boundaries_in_Obstacle:
        pytest.skip("Some boundaries in Obstacle are still not available for "
                    "cuda_native (probably AntiBounceBackOutlet)")
    flow.mask[1, 1] = 1
    collision = BGKCollision(1.0)
    simulation = Simulation(flow, collision, [])
    assert simulation.no_streaming_mask.any()
    assert simulation.no_collision_mask.any()
