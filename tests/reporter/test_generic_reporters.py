from tests.common import *

@pytest.mark.parametrize("Observable", [Enstrophy,
                                        EnergySpectrum,
                                        MaximumVelocity,
                                        IncompressibleKineticEnergy,
                                        Mass])
@pytest.mark.parametrize("Case", [[32]*2, [32]*3])
def test_generic_reporters(Observable, Case, fix_configuration):
    device, dtype, use_native = fix_configuration
    context = Context(device=device, dtype=dtype, use_native=use_native)
    flow = TaylorGreenVortex(context, Case, 10000, 0.05)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    reporter = ObservableReporter(Observable(flow),
                                  interval=1,
                                  out=None)
    simulation = Simulation(flow, collision, [reporter])
    simulation(2)
    values = np.asarray(reporter.out)
    if Observable is EnergySpectrum:
        assert values[1, 2:] == pytest.approx(values[0, 2:], rel=0.0,
                                              abs=values[0, 2:].sum() / 10)
    else:
        assert values[1, 2] == pytest.approx(values[0, 2], rel=0.05)