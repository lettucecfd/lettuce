from tests.common import *

@pytest.mark.parametrize("flowname", flow_by_name.keys())
def test_energy_spectrum(tmpdir, flowname):
    context = Context(device=torch.device('cpu'))
    IncompressibleFlow, stencil = flow_by_name[flowname]
    if IncompressibleFlow is CouetteFlow2D:
        pytest.skip("CouetteFlow2D has nan energy spectrum")
    stencil = stencil() if callable(stencil) else stencil
    flow = IncompressibleFlow(context=context, resolution=[20] * stencil.d,
                              reynolds_number=1600,
                              mach_number=0.01,
                              stencil=stencil)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    spectrum = context.convert_to_ndarray(EnergySpectrum(flow)())
    energy = IncompressibleKineticEnergy(flow)().item()

    if Flow == DecayingTurbulence:
        # check that the reported spectrum agrees with the spectrum used for initialization
        ek_ref, _ = flow.energy_spectrum
        assert (spectrum == pytest.approx(ek_ref, rel=0.0, abs=0.1))
    if Flow == TaylorGreenVortex:
        # check that flow has only one mode
        ek_max = sorted(spectrum, reverse=True)
        assert ek_max[0] * 1e-5 > ek_max[1]
    assert (energy == pytest.approx(np.sum(spectrum), rel=0.1, abs=0.0))