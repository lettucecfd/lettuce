def test_readme():
    """Whenever you have to change this test, the example in README.rst has to change, too.
    Note differences in the device + number of steps.
    """

    import torch
    from lettuce import BGKCollision, StandardStreaming, Lattice, D2Q9, TaylorGreenVortex2D, Simulation

    device = "cpu"
    dtype = torch.float32

    lattice = Lattice(D2Q9, device, dtype)
    flow = TaylorGreenVortex2D(resolution=256, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    mlups = simulation.step(num_steps=1)

    print("Performance in MLUPS:", mlups)
