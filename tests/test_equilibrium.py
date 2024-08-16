"""
Tests for equilibria
"""

from tests.common import *


@pytest.mark.parametrize("fix_equilibrium", [QuadraticEquilibrium])
def test_equilibrium_conserves_mass(fix_equilibrium, fix_device, fix_dtype,
                                    fix_stencil):
    context = Context(device=fix_device, dtype=fix_dtype, use_native=False)
    flow = TestFlow(context=context,
                    resolution=[16] * fix_stencil.d,
                    reynolds_number=100,
                    mach_number=0.1,
                    stencil=fix_stencil)
    equilibrium = fix_equilibrium()
    feq = equilibrium(flow)
    assert (flow.rho(feq).cpu().numpy()
            == pytest.approx(flow.rho().cpu().numpy()))


@pytest.mark.parametrize("fix_equilibrium", [QuadraticEquilibrium])
def test_equilibrium_conserves_momentum(fix_equilibrium, fix_device, fix_dtype,
                                    fix_stencil):
    context = Context(device=fix_device, dtype=fix_dtype, use_native=False)
    flow = TestFlow(context=context,
                    resolution=[16] * fix_stencil.d,
                    reynolds_number=100,
                    mach_number=0.1,
                    stencil=fix_stencil)
    equilibrium = fix_equilibrium()
    feq = equilibrium(flow)
    assert (flow.j(feq).cpu().numpy()
            == pytest.approx(flow.j().cpu().numpy(), abs=1e-6))
