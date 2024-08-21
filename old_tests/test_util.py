import numpy as np
import torch
from lettuce import (
    Lattice, D2Q9, D3Q27, TaylorGreenVortex2D,
    TaylorGreenVortex3D, torch_gradient, grid_fine_to_coarse,
    BGKCollision, Simulation, StandardStreaming
)
from lettuce.util import pressure_poisson
import pytest


@pytest.mark.parametrize("Stencil", [D2Q9, D3Q27])
def test_pressure_poisson(dtype_device, Stencil):
    dtype, device = dtype_device
    lattice = Lattice(Stencil(), device, dtype)
    flow_class = TaylorGreenVortex2D if Stencil is D2Q9 else TaylorGreenVortex3D
    flow = flow_class(resolution=32, reynolds_number=100, mach_number=0.05, lattice=lattice)
    p0, u = flow.initial_pu(flow.grid)
    u = flow.units.convert_velocity_to_lu(lattice.convert_to_tensor(u))
    rho0 = flow.units.convert_pressure_pu_to_density_lu(lattice.convert_to_tensor(p0))
    rho = pressure_poisson(flow.units, u, torch.ones_like(rho0))
    pfinal = flow.units.convert_density_lu_to_pressure_pu(rho).cpu().numpy()
    print(p0.max(), p0.min(), p0.mean(), pfinal.max(), pfinal.min(), pfinal.mean())
    print((p0 - pfinal).max(), (p0 - pfinal).min())
    assert p0 == pytest.approx(pfinal, rel=0.0, abs=0.05)
