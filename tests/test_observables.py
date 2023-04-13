import pytest

import numpy as np
from lettuce import Lattice, D2Q9, D3Q27, BGKCollision, DecayingTurbulence, StandardStreaming, Simulation
from lettuce.flows.taylorgreen import TaylorGreenVortex3D, TaylorGreenVortex2D
from lettuce.observables import EnergySpectrum, IncompressibleKineticEnergy


@pytest.mark.parametrize("Flow",
                         [TaylorGreenVortex2D, TaylorGreenVortex3D, 'DecayingTurbulence2D', 'DecayingTurbulence3D'])
def test_energy_spectrum(tmpdir, Flow):
    lattice = Lattice(D2Q9, device='cpu')
    if Flow == TaylorGreenVortex3D or Flow == 'DecayingTurbulence3D':
        lattice = Lattice(D3Q27, device='cpu')
    if Flow == 'DecayingTurbulence2D' or Flow == 'DecayingTurbulence3D':
        Flow = DecayingTurbulence
    flow = Flow(resolution=20, reynolds_number=1600, mach_number=0.01, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, collision, streaming)
    spectrum = lattice.convert_to_numpy(EnergySpectrum(lattice, flow)(simulation.f))
    energy = IncompressibleKineticEnergy(lattice, flow)(simulation.f).item()

    if Flow == DecayingTurbulence:
        # check that the reported spectrum agrees with the spectrum used for initialization
        ek_ref, _ = flow.energy_spectrum
        assert (spectrum == pytest.approx(ek_ref, rel=0.0, abs=0.1))
    if Flow == TaylorGreenVortex2D or Flow == TaylorGreenVortex3D:
        # check that flow has only one mode
        ek_max = sorted(spectrum, reverse=True)
        assert ek_max[0] * 1e-5 > ek_max[1]
    assert (energy == pytest.approx(np.sum(spectrum), rel=0.1, abs=0.0))
