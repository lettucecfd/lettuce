from tests.common import *
import matplotlib.pyplot as plt

@pytest.mark.parametrize("ForceType", [Guo, ShanChen])
def test_force(ForceType, fix_device):
    if fix_device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    context = Context(device=fix_device, dtype=torch.float64, use_native=False)
    # TODO cuda_native Fails here, probably because this test requires uneven
    #  grid points
    flow = PoiseuilleFlow2D(context=context,
                            resolution=17,
                            reynolds_number=1,
                            mach_number=0.02,
                            initialize_with_zeros=True)
    acceleration_lu = flow.units.convert_acceleration_to_lu(flow.acceleration)
    force = ForceType(flow=flow,
                      tau=flow.units.relaxation_parameter_lu,
                      acceleration=acceleration_lu)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu,
                             force=force)
    report_interval = 100
    errorreporter = ErrorReporter(flow.analytic_solution,
                                  interval=report_interval,
                                  out=None)
    simulation = Simulation(flow=flow,
                            collision=collision,
                            reporter=[errorreporter])
    simulation(1000)

    # compare with reference solution
    u_sim = flow.u(acceleration=acceleration_lu)
    u_sim = flow.units.convert_velocity_to_pu(u_sim)
    _, u_ref = flow.analytic_solution()

    fluidnodes = torch.eq(simulation.no_collision_mask, 0)
    # np.where(np.logical_not(flow.boundaries[0].mask.cpu()))

    print()
    print(("{:>15} " * 3).format("iterations", "error (u)", "error (p)"))
    for i in range(len(errorreporter.out)):
        error_u, error_p = np.abs(errorreporter.out[i]).tolist()
        print(f"{i * report_interval:15} {error_u:15.2e} {error_p:15.2e}")

    for dim in range(2):
        u_sim_i = context.convert_to_ndarray(u_sim[dim][fluidnodes])
        u_ref_i = context.convert_to_ndarray(u_ref[dim][fluidnodes])
        assert u_sim_i.max() == pytest.approx(u_ref_i.max(),
                                              rel=0.01)
        assert u_sim_i == pytest.approx(u_ref_i,
                                        rel=None,
                                        abs=0.01 * u_ref[0].max())
