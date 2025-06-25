from tests.conftest import *


def moment_dims_params():
    from itertools import product
    for stencil in stencil_params():
        for p in product([1, 16], repeat=stencil.d):
            yield stencil, list(p)


def moment_dims_ids():
    buffer = []
    for stencil, dims in moment_dims_params():
        buffer.append(f"{stencil.__class__.__name__}-MomentDims"
                      f"{'x'.join([str(d) for d in dims])}")
    return buffer


@pytest.fixture(params=moment_dims_params(), ids=moment_dims_ids())
def fix_stencil_x_moment_dims(request):
    return request.param


class DummyEquilibriumBoundary(EquilibriumBoundaryPU):

    # def make_no_collision_mask(self, shape: List[int], context: 'Context'
    #                            ) -> Optional[torch.Tensor]:
    #     m = context.zero_tensor(shape, dtype=bool)
    #     m[..., :1] = True
    #     return m

    def make_no_streaming_mask(self, shape: List[int], context: 'Context') -> Optional[torch.Tensor]:
        return context.one_tensor(shape, dtype=bool)



def test_equilibrium_boundary_pu_algorithm(fix_stencil, fix_configuration):
    """
    Test for the equilibrium boundary algorithm. This test verifies that the
    algorithm correctly computes the
    equilibrium outlet pressure by comparing its output to manually calculated
    equilibrium values.
    """

    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)

    pressure = 0.01
    velocity = context.convert_to_tensor([0.2] * fix_stencil.d)

    class DummyEQBC(TestFlow):
        @property
        def boundaries(self) -> List['Boundary']:
            m = self.context.zero_tensor(self.resolution, dtype=bool)
            m[..., :1] = True
            return [DummyEquilibriumBoundary(context=self.context, flow=self,
                                             mask=m, velocity=velocity,
                                             pressure=pressure)]

    flow_1 = DummyEQBC(context, resolution=fix_stencil.d * [16],
                       reynolds_number=1, mach_number=0.1, stencil=fix_stencil)
    flow_2 = TestFlow(context, resolution=16, reynolds_number=1,
                      mach_number=0.1, stencil=fix_stencil)

    simulation = Simulation(flow=flow_1, collision=NoCollision(), reporter=[])
    simulation(num_steps=1)

    # manually calculate the forced feq

    rho = flow_2.units.convert_pressure_pu_to_density_lu(pressure)
    u = flow_2.units.convert_velocity_to_lu(velocity)

    feq = flow_2.equilibrium(flow_2, rho, u)

    # apply manually calculated feq to f
    flow_2.f[..., :1] = torch.einsum("q...,q...->q...",
                                     [feq, torch.ones_like(
                                         flow_2.f)])[..., :1]

    assert context.convert_to_ndarray(flow_1.f) == pytest.approx(
        context.convert_to_ndarray(flow_2.f))


def test_equilibrium_boundary_pu_tgv(fix_stencil, fix_configuration):
    if fix_stencil.d not in [2, 3]:
        pytest.skip("TGV Test can only be done in 2D or 3D.")
    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)

    class DummyEQBC(TaylorGreenVortex):
        @property
        def boundaries(self):
            # u = self.context.one_tensor(self.stencil.d) * 0.1
            u = self.context.one_tensor(self.stencil.d) * 0.1
            p = 0  # self.context.zero_tensor([1, 1, 1])
            m = self.context.zero_tensor(self.resolution, dtype=bool)
            m[..., :1] = True
            return [DummyEquilibriumBoundary(
                self.context, self, m, u, p)]

    flow_1 = DummyEQBC(context, resolution=16, reynolds_number=1,
                       mach_number=0.1, stencil=fix_stencil)
    flow_2 = DummyTGV(context, resolution=16, reynolds_number=1,
                      mach_number=0.1, stencil=fix_stencil)

    simulation = Simulation(flow=flow_1, collision=NoCollision(), reporter=[])
    simulation(num_steps=1)

    pressure = 0
    velocity = 0.1 * np.ones(flow_2.stencil.d)

    feq = flow_2.equilibrium(
        flow_2,
        context.convert_to_tensor(
            flow_2.units.convert_pressure_pu_to_density_lu(pressure)),
        context.convert_to_tensor(
            flow_2.units.convert_velocity_to_lu(velocity))
    )
    flow_2.f[..., :1] = torch.einsum("q,q...->q...", feq,
                                     torch.ones_like(flow_2.f))[..., :1]

    assert flow_1.f.cpu().numpy() == pytest.approx(flow_2.f.cpu().numpy())


def test_equilibrium_boundary_pu_native(fix_stencil_x_moment_dims, fix_dtype):
    if not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available on this machine.")
    stencil, moment_dims = fix_stencil_x_moment_dims

    context_native = Context(device=torch.device('cuda'), dtype=fix_dtype,
                             use_native=True)
    context_cpu = Context(device=torch.device('cpu'), dtype=fix_dtype,
                          use_native=False)

    velocity = 0.2 * np.ones([stencil.d] + moment_dims)
    pressure = 0.02 * np.ones([1] + moment_dims)

    class DummyEQBC(TestFlow):
        @property
        def boundaries(self) -> List[Boundary]:
            m = self.context.zero_tensor(self.resolution, dtype=bool)
            m[..., :1] = True
            return [DummyEquilibriumBoundary(flow=self, context=self.context,
                                             mask=m,
                                             velocity=self.context.
                                             convert_to_tensor(velocity),
                                             pressure=self.context.
                                             convert_to_tensor(pressure))]

    flow_native = DummyEQBC(context_native, resolution=16, reynolds_number=1,
                            mach_number=0.1, stencil=stencil)
    flow_cpu = DummyEQBC(context_cpu, resolution=16, reynolds_number=1,
                         mach_number=0.1, stencil=stencil)

    simulation_native = Simulation(flow=flow_native, collision=NoCollision(),
                                   reporter=[])
    simulation_cpu = Simulation(flow=flow_cpu, collision=NoCollision(),
                                reporter=[])

    simulation_native(num_steps=1)
    simulation_cpu(num_steps=1)

    assert flow_cpu.f.cpu().numpy() == pytest.approx(
        flow_native.f.cpu().numpy())
