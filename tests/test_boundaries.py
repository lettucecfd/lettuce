"""
Test boundary conditions.
"""

from lettuce import BounceBackBoundary, EquilibriumBoundaryPU, UnitConversion

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

