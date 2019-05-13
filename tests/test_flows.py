
import pytest
from lettuce import TaylorGreenVortex2D, CouetteFlow2D
from lettuce import Lattice, LatticeOfVector, D2Q9, Simulation, BGKCollision, StandardStreaming


@pytest.mark.parametrize("IncompressibleFlow", [TaylorGreenVortex2D, CouetteFlow2D])
@pytest.mark.parametrize("Ltc", [Lattice, LatticeOfVector])
def test_flow(IncompressibleFlow, Ltc, dtype_device):
    dtype, device = dtype_device
    lattice = Ltc(D2Q9, dtype=dtype, device=device)
    flow = IncompressibleFlow(16, 1, 0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.step(1)
