"""
Test functions for collision models and related functions.
"""

import pytest

from lettuce import *

from copy import copy


@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_conserves_mass(Collision, f_all_lattices):
    f, lattice = f_all_lattices
    f_old = copy(f)
    collision = Collision(lattice, 0.51)
    f = collision(f)
    assert lattice.rho(f).cpu().numpy() == pytest.approx(lattice.rho(f_old).cpu().numpy())


@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_conserves_momentum(Collision, f_all_lattices):
    f, lattice = f_all_lattices
    f_old = copy(f)
    collision = Collision(lattice, 0.51)
    f = collision(f)
    assert lattice.j(f).cpu().numpy() == pytest.approx(lattice.j(f_old).cpu().numpy(), abs=1e-5)


@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_fixpoint_2x(Collision, f_all_lattices):
    f, lattice = f_all_lattices
    f_old = copy(f)
    collision = Collision(lattice, 0.5)
    f = collision(collision(f))
    assert f.cpu().numpy() == pytest.approx(f_old.cpu().numpy(), abs=1e-5)


