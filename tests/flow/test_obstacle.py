from tests.conftest import *


@pytest.mark.parametrize("stencil2d3d", [D2Q9(), D3Q27()])
def test_divergence(stencil2d3d, fix_configuration):
    device, dtype, use_native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=use_native)

    nx = 32
    ny = 16
    nz = 16

    resolution = [nx, ny, nz] if stencil2d3d.d == 3 else [nx, ny]
    mask = np.zeros(resolution)
    if stencil2d3d.d == 3:
        mask[3:6, 3:6, :] = 1
    else:
        mask[3:6, 3:6] = 1
    flow = Obstacle(context=context,
                    resolution=resolution,
                    reynolds_number=100,
                    mach_number=0.1,
                    domain_length_x=3)
    all_native_boundaries_in_Obstacle = sum([
        _.native_available() for _ in flow.boundaries]) == flow.boundaries
    if use_native and not all_native_boundaries_in_Obstacle:
        pytest.skip("Some boundaries in Obstacle are still not available for "
                    "cuda_native (probably AntiBounceBackOutlet)")
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    flow.mask = mask != 0
    simulation = Simulation(flow, collision, [])
    simulation(2)
