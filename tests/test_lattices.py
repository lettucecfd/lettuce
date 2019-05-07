"""
Tests for lattices.
"""

import pytest


def test_quadratic_equilibrium_conserves_momentum(f_lattice):
    f, lattice = f_lattice
    feq = lattice.quadratic_equilibrium(rho = lattice.rho(f), u = lattice.u(f))
    assert lattice.rho(feq).numpy() == pytest.approx(lattice.rho(f).numpy())


def test_quadratic_equilibrium_conserves_momentum(f_lattice):
    f, lattice = f_lattice
    feq = lattice.quadratic_equilibrium(rho = lattice.rho(f), u = lattice.u(f))
    assert lattice.j(feq).numpy() == pytest.approx(lattice.j(f).numpy())
