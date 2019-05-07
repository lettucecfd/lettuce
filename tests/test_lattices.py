"""
Tests for lattices.
"""

import pytest


def test_quadratic_equilibrium_conserves_momentum(f_lattice):
    f, lattice = f_lattice
    feq = lattice.quadratic_equilibrium(rho = lattice.rho(f), u = lattice.u(f))
    assert lattice.rho(feq).cpu().numpy() == pytest.approx(lattice.rho(f).cpu().numpy())


def test_quadratic_equilibrium_conserves_momentum(f_lattice):
    f, lattice = f_lattice
    feq = lattice.quadratic_equilibrium(rho = lattice.rho(f), u = lattice.u(f))
    assert lattice.j(feq).cpu().numpy() == pytest.approx(lattice.j(f).cpu().numpy())
