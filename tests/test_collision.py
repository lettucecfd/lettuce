"""
Test functions for collision models and related functions.
"""

import numpy as np
import torch
import pytest
from copy import copy

from lettuce import *


@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_conserves_mass(Collision, f_lattice):
    f, lattice = f_lattice
    f_old = copy(f)
    collision = Collision(lattice, 0.51)
    f = collision(f)
    assert lattice.rho(f).numpy() == pytest.approx(lattice.rho(f_old).numpy())


@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_conserves_momentum(Collision, f_lattice):
    f, lattice = f_lattice
    f_old = copy(f)
    collision = Collision(lattice, 0.51)
    f = collision(f)
    assert lattice.j(f).numpy() == pytest.approx(lattice.j(f_old).numpy(), abs=1e-5)


@pytest.mark.parametrize("Collision", [BGKCollision])
def test_collision_fixpoint_2x(Collision, f_lattice):
    f, lattice = f_lattice
    f_old = copy(f)
    collision = Collision(lattice, 0.5)
    f = collision(collision(f))
    assert f.numpy() == pytest.approx(f_old.numpy(), abs=1e-5)

