
import pytest
import numpy as np
from lettuce import (
    EquivariantNeuralCollision, Lattice,
    get_default_moment_transform, LettuceException,
    D1Q3, D2Q9, D2Q9Dellar, LettuceInvalidNetworkOutput,
    GConv, GConvPermutation, EquivariantNet
)
import torch


@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("in_channels", [1, 3])
def test_gconv_equivariance(symmetry_group, dtype_device, in_channels, out_channels):
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    try:
        moments = get_default_moment_transform(lattice)
    except LettuceException:
        pytest.skip(f"No moment transform for {lattice.stencil}")
    f = lattice.convert_to_tensor(np.random.random([in_channels, lattice.Q] + [3] * lattice.D))
    m = moments.transform(f)
    conv = GConv(
        in_channels,
        out_channels,
        symmetry_group.moment_action(moments),
        symmetry_group.inverse_moment_action(moments)
    )
    conv.to(dtype=dtype, device=device)

    def apply(rep, p):
        xyz = "xyz"[:lattice.D]
        return torch.einsum(f"ij, fj{xyz}->fj{xyz}", rep, p)

    for action in symmetry_group.moment_action(moments):
        assert torch.allclose(
            conv(apply(action, m)),
            apply(action, conv(m)),
            atol=1e-3 if dtype == torch.float32 else 1e-5
        )


def test_gconv_permutation_equivariance(symmetry_group, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    f = lattice.convert_to_tensor(np.random.random([1] + [lattice.Q] + [3] * lattice.D))
    conv = GConvPermutation(1, 1, symmetry_group.permutations, symmetry_group.inverse_permutations)
    conv.to(dtype=dtype, device=device)
    for p in symmetry_group.permutations:
        assert torch.allclose(
            conv(f[:, p]),
            conv(f)[:, p],
            atol=1e-3 if dtype == torch.float32 else 1e-5
        )


def test_equivariant_net(symmetry_group, dtype_device):
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    try:
        moments = get_default_moment_transform(lattice)
    except LettuceException:
        pytest.skip(f"No moment transform for {lattice.stencil}")
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    m = moments.transform(f)
    m = m.moveaxis(0, lattice.D)
    net = torch.nn.Sequential(torch.nn.Linear(lattice.Q, 23), torch.nn.ReLU(), torch.nn.Linear(23, lattice.Q))
    equi = EquivariantNet(
        net=net,
        group_actions=symmetry_group.moment_action(moments),
        inverse_group_actions=symmetry_group.inverse_moment_action(moments),
    )
    equi.to(dtype=dtype, device=device)

    def apply(rep, p):
        xyz = "xyz"[:lattice.D]
        return torch.einsum(f"ij, {xyz}j->{xyz}i", rep, p)

    for action in symmetry_group.moment_action(moments):
        assert torch.allclose(
            equi(apply(action, m)),
            apply(action, equi(m)),
            atol=1e-3 if dtype == torch.float32 else 1e-5
        )


def test_equivariant_net_selected_moments(symmetry_group, dtype_device):
    if symmetry_group.stencil == D1Q3:
        pytest.skip("Too few moments in D1Q3")
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    try:
        moments = get_default_moment_transform(lattice)
    except LettuceException:
        pytest.skip(f"No moment transform for {lattice.stencil}")
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    m = moments.transform(f)
    m = m.moveaxis(0, lattice.D)
    moment_orders = np.array([sum(name.count(x) for x in "xyz") for name in moments.names])
    in_indices = np.where(moment_orders <= 2)[0]
    out_indices = np.where(moment_orders > 2)[0]
    #net = torch.nn.Linear(len(in_indices), len(out_indices))
    #in_indices = np.arange(1 + lattice.D)
    #out_indices = np.arange(1 + lattice.D, lattice.Q)
    net = torch.nn.Sequential(
        torch.nn.Linear(len(in_indices), 23),
        torch.nn.ReLU(),
        torch.nn.Linear(23, len(out_indices))
    )
    equi = EquivariantNet(
        net=net,
        group_actions=symmetry_group.moment_action(moments),
        inverse_group_actions=symmetry_group.inverse_moment_action(moments),
        in_indices=in_indices,
        out_indices=out_indices
    )
    equi.to(dtype=dtype, device=device)

    def apply(rep, p):
        xyz = "xyz"[:lattice.D]
        return torch.einsum(f"ij, {xyz}j->{xyz}i", rep, p)

    for action in symmetry_group.moment_action(moments):
        assert torch.allclose(
            equi(apply(action[in_indices][..., in_indices], m[..., in_indices])),
            apply(action[out_indices][..., out_indices], equi(m[..., in_indices])),
            atol=1e-3 if dtype == torch.float32 else 1e-5
        )


@pytest.mark.slow
def test_equivariant_mrt(symmetry_group, dtype_device):
    if symmetry_group.stencil == D1Q3:
        pytest.skip("Too few moments in D1Q3")
    dtype, device = dtype_device
    lattice = Lattice(symmetry_group.stencil, dtype=dtype, device=device)
    try:
        moment_transform = get_default_moment_transform(lattice)
    except LettuceException:
        pytest.skip()
    moment_orders = np.array([sum(name.count(x) for x in "xyz") for name in moment_transform.names])
    net = torch.nn.Linear((moment_orders <= 2).sum(), (moment_orders > 2).sum())
    collision = EquivariantNeuralCollision(0.7, net, moment_transform)
    f = lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D))
    for p in symmetry_group.permutations:
        print(torch.norm(collision(f[p]) - collision(f)[p]).item())
        print(collision(f[p])/(collision(f)[p]))
        assert torch.allclose(
            collision(f[p]),
            collision(f)[p],
            atol=1e-2 if dtype == torch.float32 else 1e-4
        )
