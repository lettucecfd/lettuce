"""
Test functions for collision models and related functions.
"""
from copy import copy

import pytest

from lettuce import *


@pytest.mark.parametrize("Collision", [BGKCollision, KBCCollision2D, KBCCollision3D, TRTCollision, RegularizedCollision, SmagorinskyCollision])
def test_collision_conserves_mass(Collision, f_all_lattices):
    f, lattice = f_all_lattices
    if ((Collision == KBCCollision2D and lattice.stencil != D2Q9) or ((Collision == KBCCollision3D and lattice.stencil != D3Q27 ))) :
        pytest.skip()
    f_old = copy(f)
    collision = Collision(lattice, 0.51)
    f = collision(f)
    assert lattice.rho(f).cpu().numpy() == pytest.approx(lattice.rho(f_old).cpu().numpy())


@pytest.mark.parametrize("Collision", [BGKCollision, KBCCollision2D, KBCCollision3D, TRTCollision, RegularizedCollision, SmagorinskyCollision])
def test_collision_conserves_momentum(Collision, f_all_lattices):
    f, lattice = f_all_lattices
    if ((Collision == KBCCollision2D and lattice.stencil != D2Q9) or ((Collision == KBCCollision3D and lattice.stencil != D3Q27 ))) :
        pytest.skip()
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


@pytest.mark.parametrize("Collision", [BGKCollision, TRTCollision, KBCCollision2D, KBCCollision3D, RegularizedCollision])
def test_collision_relaxes_shear_moments(Collision, f_all_lattices):
    """checks whether the collision models relax the shear moments according to the prescribed relaxation time"""
    f, lattice = f_all_lattices
    if ((Collision == KBCCollision2D and lattice.stencil != D2Q9) or ((Collision == KBCCollision3D and lattice.stencil != D3Q27 ))) :
        pytest.skip()
    rho = lattice.rho(f)
    u = lattice.u(f)
    feq = lattice.equilibrium(rho, u)
    shear_pre = lattice.shear_tensor(f)
    shear_eq_pre = lattice.shear_tensor(feq)
    tau = 0.6
    coll = Collision(lattice, tau)
    f_post = coll(f)
    shear_post = lattice.shear_tensor(f_post)
    assert shear_post.cpu().numpy() == pytest.approx((shear_pre - 1 / tau * (shear_pre - shear_eq_pre)).cpu().numpy(),
                                                     abs=1e-5)


@pytest.mark.parametrize("Collision", [KBCCollision2D, KBCCollision3D])
def test_collision_optimizes_pseudo_entropy(Collision, f_all_lattices):
    "checks if the pseudo-entropy of the KBC collision model is at least higher than the BGK pseudo-entropy"
    f, lattice = f_all_lattices
    if ((Collision == KBCCollision2D and lattice.stencil != D2Q9) or ((Collision == KBCCollision3D and lattice.stencil != D3Q27 ))) :
        pytest.skip()
    tau = 0.5003
    coll_kbc = Collision(lattice, tau)
    coll_bgk = BGKCollision(lattice, tau)
    f_kbc = coll_kbc(f)
    f_bgk = coll_bgk(f)
    entropy_kbc = lattice.pseudo_entropy_local(f_kbc)
    entropy_bgk = lattice.pseudo_entropy_local(f_bgk)
    assert (entropy_bgk.cpu().numpy() < entropy_kbc.cpu().numpy()).all()


@pytest.mark.parametrize("Transform", [D2Q9Lallemand, D2Q9Dellar])
def test_collision_fixpoint_2x_MRT(Transform, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(D2Q9, device=device, dtype=dtype)
    np.random.seed(1) # arbitrary, but deterministic
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3]*lattice.D))
    f_old = copy(f)
    collision = MRTCollision(lattice, Transform(lattice), np.array([0.5]*9))
    f = collision(collision(f))
    print(f.cpu().numpy(), f_old.cpu().numpy())
    assert f.cpu().numpy() == pytest.approx(f_old.cpu().numpy(), abs=1e-5)

