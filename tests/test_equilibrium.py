"""
Tests for equilibria
"""

import pytest
from lettuce.equilibrium import *


@pytest.mark.parametrize("Equilibrium", [QuadraticEquilibrium])
def test_equilibrium_conserves_mass(f_all_lattices, Equilibrium):
    f, lattice = f_all_lattices
    equilibrium = Equilibrium(lattice)
    feq = equilibrium(rho=lattice.rho(f), u=lattice.u(f))
    assert lattice.rho(feq).cpu().numpy() == pytest.approx(lattice.rho(f).cpu().numpy())


@pytest.mark.parametrize("Equilibrium", [QuadraticEquilibrium])
def test_equilibrium_conserves_momentum(f_all_lattices, Equilibrium):
    f, lattice = f_all_lattices
    equilibrium = Equilibrium(lattice)
    feq = equilibrium(rho=lattice.rho(f), u=lattice.u(f))
    assert lattice.j(feq).cpu().numpy() == pytest.approx(lattice.j(f).cpu().numpy(), abs=1e-6)
