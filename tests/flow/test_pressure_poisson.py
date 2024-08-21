import pytest

from tests.common import *


@pytest.mark.parametrize("Stencil", [D2Q9, D3Q27])
def test_pressure_poisson(fix_configuration, Stencil):
    if Stencil == D3Q27:
        pytest.skip("D3Q27 pressure_poisson does not work.")
    device, dtype, use_native = fix_configuration
    context = Context(device, dtype, use_native)
    flow = TaylorGreenVortex(context=context,
                             resolution=32,
                             reynolds_number=100,
                             mach_number=0.05,
                             stencil=Stencil)
    p0, u = flow.initial_pu()
    u = flow.units.convert_velocity_to_lu(u)
    rho0 = flow.units.convert_pressure_pu_to_density_lu(p0)
    rho = pressure_poisson(flow.units, u, torch.ones_like(rho0))
    pfinal = flow.units.convert_density_lu_to_pressure_pu(rho).cpu().numpy()
    print(p0.max(), p0.min(), p0.mean(), pfinal.max(), pfinal.min(), pfinal.mean())
    print((p0 - pfinal).max(), (p0 - pfinal).min())
    assert p0 == pytest.approx(pfinal, rel=0.0, abs=0.05)
