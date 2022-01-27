"""
Fixtures for unit tests.
"""
import pytest

import numpy as np
import torch

from lettuce import (
    stencils, Stencil, get_subclasses, Transform, Lattice, moments
)

STENCILS = list(get_subclasses(Stencil, stencils))
TRANSFORMS = list(get_subclasses(Transform, moments))


@pytest.fixture(
    params=["cpu", pytest.param(
        "cuda:0", marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available.")
    )])
def device(request):
    """Run a test case for all available devices."""
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
# not testing torch.float16 (half precision is not precise enough)
def dtype_device(request, device):
    """Run a test case for all available devices and data types available on the device."""
    if device == "cpu" and request.param == torch.float16:
        pytest.skip("Half precision is only available on GPU.")
    if device == "cuda:0" and request.param == torch.float32:
        pytest.skip("Tolerances are not yet adapted to CUDA single precision.")
    return request.param, device


@pytest.fixture(params=STENCILS)
def stencil(request):
    """Run a test for all stencils."""
    return request.param


@pytest.fixture(params=STENCILS)
def lattice(request, dtype_device):
    """Run a test for all lattices (all stencils, devices and data types available on the device.)"""
    dtype, device = dtype_device
    return Lattice(request.param, device=device, dtype=dtype)


@pytest.fixture()
def f_lattice(lattice):
    """Run a test for all lattices; return a grid with 3^D sample distribution functions alongside the lattice."""
    np.random.seed(1)  # arbitrary, but deterministic
    return lattice.convert_to_tensor(np.random.random([lattice.Q] + [3] * lattice.D)), lattice


@pytest.fixture(params=[Lattice])
def f_all_lattices(request, lattice):
    """Run a test for all lattices and lattices-of-vector;
    return a grid with 3^D sample distribution functions alongside the lattice.
    """
    np.random.seed(1)
    f = np.random.random([lattice.Q] + [3] * lattice.D)
    Ltc = request.param
    ltc = Ltc(lattice.stencil, lattice.device, lattice.dtype)
    return ltc.convert_to_tensor(f), ltc


@pytest.fixture(params=TRANSFORMS)
def f_transform(request, f_all_lattices):
    Transform = request.param
    f, lattice = f_all_lattices
    if lattice.stencil in Transform.supported_stencils:
        return f, Transform(lattice)
    else:
        pytest.skip("Stencil not supported for this transform.")
