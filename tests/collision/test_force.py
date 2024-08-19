from tests.common import *
import matplotlib.pyplot as plt

@pytest.mark.parametrize("ForceType", [Guo, ShanChen])
def test_force(ForceType, fix_device):
    if fix_device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    context = Context(device=fix_device, dtype=torch.float64, use_native=False)
    # TODO use_native Fails here
    flow = PoiseuilleFlow2D(context=context,
                            resolution=16,
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

    u_sim = context.convert_to_ndarray(u_sim)
    u_ref = context.convert_to_ndarray(u_ref)

    fluidnodes = context.convert_to_ndarray(
        torch.eq(simulation.no_collision_mask, 0))
    #np.where(np.logical_not(flow.boundaries[0].mask.cpu()))

    fig, ax = plt.subplots(1, 3)
    fig.suptitle(f"ForceType: {ForceType.__name__}")
    p1 = ax[0].imshow(u_sim[0])
    plt.colorbar(mappable=p1, ax=ax[0])
    ax[0].set_title("Simulation")
    p2 = ax[1].imshow(u_ref[0])
    plt.colorbar(mappable=p2, ax=ax[1])
    ax[1].set_title("Reference")
    p3 = ax[2].imshow(u_ref[0]-u_sim[0])
    plt.colorbar(mappable=p3, ax=ax[2])
    ax[2].set_title("Difference")
    plt.show()

    print()
    print(("{:>15} " * 3).format("iterations", "error (u)", "error (p)"))
    for i in range(len(errorreporter.out)):
        error_u, error_p = np.abs(errorreporter.out[i]).tolist()
        print(f"{i*report_interval:15} {error_u:15.2e} {error_p:15.2e}")

    for dim in range(2):
        assert u_sim[dim].max() == pytest.approx(u_ref[dim].max(),
                                                 rel=0.01)
        assert u_sim[dim][fluidnodes] == pytest.approx(u_ref[dim][fluidnodes],
                                                 # rel=None,
                                                 abs=0.01 * u_ref[0].max())