from tests.common import *


def test_equilibrium_pressure_outlet(fix_configuration, fix_stencil):
    pytest.skip("TODO rtol is too big or simulation time too short!")
    if fix_stencil.d not in [2, 3]:
        pytest.skip("Obstacle to test EQ P outlet only implemented for 2D "
                    "and 3D")
    outlet_directions = [[0, -1],
                         [0, 1],
                         [1, 0]]
    if fix_stencil.d == 3:
        outlet_directions = [direction + [0]
                             for direction in outlet_directions]
    device, dtype, native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=native)

    class MyObstacle(Obstacle):
        @property
        def boundaries(self, *args):
            x, y, *z = self.grid
            z = z if z else None
            outlets = [EquilibriumOutletP(_, self) for _ in outlet_directions]
            inflow = [self.units.characteristic_velocity_pu, 0] \
                if self.stencil.d == 2 \
                else [self.units.characteristic_velocity_pu, 0, 0]
            return [
                EquilibriumBoundaryPU(
                    self.context,
                    np.abs(x) < 1e-6,
                    inflow
                ),
                *outlets,
                BounceBackBoundary(self.mask)
            ]

    flow = MyObstacle(context=context,
                      resolution=[32] * fix_stencil.d,
                      reynolds_number=10,
                      mach_number=0.1,
                      domain_length_x=3,
                      stencil=fix_stencil)
    all_native_boundaries_in_MyObstacle = sum([
        _.native_available() for _ in flow.boundaries]) == flow.boundaries
    if native and not all_native_boundaries_in_MyObstacle:
        pytest.skip("Some boundaries in Obstacle are still not available for "
                    "cuda_native (probably EquilibriumOutletP)")

    # mask = np.zeros_like(flow.grid[0], dtype=bool)
    # mask[10:20, 10:20] = 1
    flow.mask[10:20, 10:20] = True
    simulation = Simulation(flow,
                            RegularizedCollision(
                                flow.units.relaxation_parameter_lu),
                            [VTKReporter(1,
                                         filename_base=f"./data/output_"
                                                       f"D{fix_stencil.d}Q"
                                                       f"{fix_stencil.q}")])
    simulation(30)
    rho = flow.rho()
    u = flow.u()
    feq = flow.equilibrium(flow, torch.ones_like(rho), u)
    p = flow.units.convert_density_lu_to_pressure_pu(rho)
    zeros = torch.zeros_like(p[0, -1, :])
    # TODO rtol is too big or simulation time too short!
    assert torch.allclose(zeros, p[:, -1, :], rtol=0, atol=1e-4)
    assert torch.allclose(zeros, p[:, :, 0], rtol=0, atol=1e-4)
    assert torch.allclose(zeros, p[:, :, -1], rtol=0, atol=1e-4)
    assert torch.allclose(feq[:, -1, 1:-1], feq[:, -2, 1:-1])
