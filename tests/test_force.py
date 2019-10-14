def test_flow_2d(IncompressibleFlow, Ltc, dtype_device):
    dtype, device = dtype_device
    lattice = Ltc(D2Q9, dtype=dtype, device=device)
    flow = IncompressibleFlow(16, 1, 0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(2)