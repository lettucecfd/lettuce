from tests.common import *


def moment_dims_params():
    from itertools import product
    for stencil in stencil_params():
        for p in product([1, 16], repeat=stencil.d):
            yield stencil, list(p)


def moment_dims_ids():
    buffer = []
    for stencil, dims in moment_dims_params():
        buffer.append(f"{stencil.__class__.__name__}-MomentDims{'x'.join([str(d) for d in dims])}")
    return buffer


@pytest.fixture(params=moment_dims_params(), ids=moment_dims_ids())
def fix_stencil_x_moment_dims(request):
    return request.param


class TestEquilibriumBoundary(EquilibriumBoundaryPU):

    def make_no_collision_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        m = context.zero_tensor(shape, dtype=bool)
        m[..., :1] = True
        return m

    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        return context.one_tensor(shape, dtype=bool)


def test_equilibrium_boundary_pu_algorithm(fix_stencil, fix_configuration):
    """
    Test for the _equilibrium _boundary algorithm. This test verifies that the algorithm correctly computes the
    _equilibrium outlet pressure by comparing its output to manually calculated _equilibrium values.
    """

    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)

    flow_1 = TestFlow(context, resolution=fix_stencil.d * [16], reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    flow_2 = TestFlow(context, resolution=fix_stencil.d * [16], reynolds_number=1, mach_number=0.1, stencil=fix_stencil)

    velocity = 0.2 * np.ones(flow_2.stencil.d)
    pressure = 0.01

    boundary = TestEquilibriumBoundary(context, velocity, pressure)
    simulation = Simulation(flow=flow_1, collision=NoCollision(), boundaries=[boundary], reporter=[])
    simulation(num_steps=1)

    # manually calculate the forced feq

    rho = flow_2.units.convert_pressure_pu_to_density_lu(context.convert_to_tensor(pressure))
    u = flow_2.units.convert_velocity_to_lu(context.convert_to_tensor(velocity))

    feq = flow_2.equilibrium(flow_2, rho, u)

    # apply manually calculated feq to f
    flow_2.f[..., :1] = torch.einsum("q,q...->q...", feq, torch.ones_like(flow_2.f))[..., :1]

    assert flow_1.f.cpu().numpy() == pytest.approx(flow_2.f.cpu().numpy())


def test_equilibrium_boundary_pu_native(fix_stencil_x_moment_dims, fix_dtype):
    stencil, moment_dims = fix_stencil_x_moment_dims

    context_native = Context(device=torch.device('cuda'), dtype=fix_dtype, use_native=True)
    context_cpu = Context(device=torch.device('cpu'), dtype=fix_dtype, use_native=False)

    flow_native = TestFlow(context_native, stencil=stencil, resolution=16, reynolds_number=1, mach_number=0.1)
    flow_cpu = TestFlow(context_cpu, stencil=stencil, resolution=16, reynolds_number=1, mach_number=0.1)

    velocity = 0.2 * np.ones([flow_cpu.stencil.d] + moment_dims)
    pressure = 0.02 * np.ones([1] + moment_dims)

    boundary_native = TestEquilibriumBoundary(context_native, velocity, pressure)
    boundary_cpu = TestEquilibriumBoundary(context_cpu, velocity, pressure)

    simulation_native = Simulation(flow=flow_native, collision=NoCollision(), boundaries=[boundary_native], reporter=[])
    simulation_cpu = Simulation(flow=flow_cpu, collision=NoCollision(), boundaries=[boundary_cpu], reporter=[])

    simulation_native(num_steps=1)
    simulation_cpu(num_steps=1)

    assert flow_cpu.f.cpu().numpy() == pytest.approx(flow_native.f.cpu().numpy())
