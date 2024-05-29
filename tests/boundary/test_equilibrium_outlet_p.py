"""
Test boundary conditions.
"""

import pytest
import torch

from lettuce import *
from lettuce.ext import *
from tests.conftest import TestFlow


# TODO: Implement native generator and test suite

def test_equilibrium_outlet_p_algorithm(stencils, configurations):
    '''
    Test for the equilibrium outlet p boundary algorithm. This test verifies that the algorithm correctly computes the
    equilibrium outlet pressure by comparing its output to manually calculated equilibrium values.
    '''
    dtype, device, native = configurations
    stencil = stencils()
    if native == "native":
        pytest.skip("TODO: native_available for equilibrium_outlet_p at the moment False")
    context = Context(device=torch.device(device), dtype=dtype, use_native=(native == "native"))

    flow = TestFlow(context, resolution=stencil.d*[16], reynolds_number=1, mach_number=0.1, stencil=stencil)
    direction = [0] * (stencil.d-1) + [1]
    boundary_cpu = EquilibriumOutletP(flow=flow, context=context, direction=direction, rho_outlet=1.2)
    f_post_boundary = boundary_cpu(flow)[..., -1]
    u_slice = [stencil.d, *flow.resolution[:stencil.d-1],1]
    rho_slice = [1,*flow.resolution[:stencil.d-1],1]
    u = flow.units.convert_velocity_to_lu(context.one_tensor(u_slice))
    rho = context.one_tensor(rho_slice) * 1.2
    reference = flow.equilibrium(flow, rho=rho, u=u)[..., 0]
    assert reference.cpu().numpy() == pytest.approx(f_post_boundary.cpu().numpy(), rel=1e-6)
