"""
Test boundary conditions.
"""

from lettuce import (
    BounceBackBoundary, EquilibriumBoundaryPU,
    UnitConversion, AntiBounceBackOutlet, D3Q27, D3Q19, D2Q9, D1Q3, Obstacle2D, Lattice,
    StandardStreaming, Simulation
)


import pytest

from copy import copy
import numpy as np
import torch


def test_bounce_back_boundary(f_lattice):
    f, lattice = f_lattice
    f_old = copy(f)
    mask = (f[0] > 0).cpu().numpy()  # will contain all points
    bounce_back = BounceBackBoundary(mask, lattice)
    f = bounce_back(f)
    assert f[lattice.stencil.opposite].cpu().numpy() == pytest.approx(f_old.cpu().numpy())


def test_bounce_back_boundary_not_applied_if_mask_empty(f_lattice):
    f, lattice = f_lattice
    f_old = copy(f)
    mask = (f[0] < 0).cpu().numpy()  # will not contain any points
    bounce_back = BounceBackBoundary(mask, lattice)
    f = bounce_back(f)
    assert f.cpu().numpy() == pytest.approx(f_old.cpu().numpy())


def test_equilibrium_boundary_pu(f_lattice):
    f, lattice = f_lattice
    mask = (f[0] > 0).cpu().numpy()  # will contain all points
    units = UnitConversion(lattice, reynolds_number=1)
    pressure = 0
    velocity = 0.1*np.ones(lattice.D)
    feq = lattice.equilibrium(
            lattice.convert_to_tensor(units.convert_pressure_pu_to_density_lu(pressure)),
            lattice.convert_to_tensor(units.convert_velocity_to_lu(velocity))
        )
    feq_field = torch.einsum("q,q...->q...", feq, torch.ones_like(f))

    eq_boundary = EquilibriumBoundaryPU(mask, lattice, units, velocity=velocity, pressure=pressure)
    f = eq_boundary(f)

    assert f.cpu().numpy() == pytest.approx(feq_field.cpu().numpy())
    #assert f.cpu().numpy() == pytest.approx(f_old.cpu().numpy())

def test_anti_bounce_back_outlet(f_lattice):
    """Compares the result of the application of the boundary to f to the result using the formula taken from page 195
    of "The lattice Boltzmann method" (2016 by Krüger et al.) if both are similar it is assumed to be working fine."""
    f, lattice = f_lattice
    #generates reference value of f using non-dynamic formula
    f_ref = f
    u = lattice.u(f)
    D = lattice.stencil.D()
    Q = lattice.stencil.Q()
    if D == 3:
        direction = [1, 0, 0]
        if Q == 27:
            u_w = u[:, -1, :, :] + 0.5 * (u[:, -1, :, :] - u[:, -2, :, :])
            for i in [1, 11, 13, 15, 17, 19, 21, 23, 25]:
                f_ref[lattice.stencil.opposite[i], -1, :, :] = - f_ref[i, -1, :, :] + lattice.stencil.w[i] * lattice.rho(f)[0, -1, :, :] * \
                    (2 + torch.einsum('c, cyz -> yz', torch.tensor(lattice.stencil.e[i], device=f.device, dtype=f.dtype), u_w) ** 2 / lattice.stencil.cs ** 4 - (torch.norm(u_w, dim=0) / lattice.stencil.cs) ** 2)
        if Q == 19:
            u_w = u[:, -1, :, :] + 0.5 * (u[:, -1, :, :] - u[:, -2, :, :])
            for i in [1, 11, 13, 15, 17]:
                f_ref[lattice.stencil.opposite[i], -1, :, :] = - f_ref[i, -1, :, :] + lattice.stencil.w[i] * lattice.rho(f)[0, -1, :, :] * \
                    (2 + torch.einsum('c, cyz -> yz', torch.tensor(lattice.stencil.e[i], device=f.device, dtype=f.dtype), u_w) ** 2 / lattice.stencil.cs ** 4 - (torch.norm(u_w, dim=0) / lattice.stencil.cs) ** 2)
    if D == 2 and Q == 9:
        direction = [1, 0]
        u_w = u[:, -1, :] + 0.5 * (u[:, -1, :] - u[:, -2, :])
        for i in [1, 5, 8]:
            f_ref[lattice.stencil.opposite[i], -1,:] = - f_ref[i, -1, :] + lattice.stencil.w[i] * lattice.rho(f)[0, -1, :] * \
                    (2 + torch.einsum('c, cy -> y', torch.tensor((lattice.stencil.e[i]), device=f.device, dtype=f.dtype), u_w) ** 2 / lattice.stencil.cs ** 4 - (torch.norm(u_w, dim=0) / lattice.stencil.cs)**2)
    if D == 1 and Q == 3:
        direction = [1]
        u_w = u[:, -1] + 0.5 * (u[:, -1] - u[:, -2])
        for i in [1]:
            f_ref[lattice.stencil.opposite[i], -1] = - f_ref[i, -1] + lattice.stencil.w[i] * lattice.rho(f)[0, -1] * \
                    (2 + torch.einsum('c, x -> x', (torch.tensor((lattice.stencil.e[i]), device=f.device, dtype=f.dtype), u_w)) ** 2 / lattice.stencil.cs ** 4 - (torch.norm(u_w, dim=0) / lattice.stencil.cs)**2)
    # generates value from actual boundary implementation
    abb_outlet = AntiBounceBackOutlet(lattice, direction)
    f = abb_outlet(f)
    assert f.cpu().numpy() == pytest.approx(f_ref.cpu().numpy())


def test_masks(dtype_device):
    """test if masks are applied from boundary conditions"""
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, dtype=dtype, device=device)
    flow = Obstacle2D(10, 5, 100, 0.1, lattice, 2)
    flow.mask[1,1] = 1
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow, lattice, None, streaming)
    assert simulation.streaming.no_stream_mask.any()
    assert simulation.no_collision_mask.any()

