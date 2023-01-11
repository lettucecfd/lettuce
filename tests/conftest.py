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
    params=["cpu",
            pytest.param("cuda:0",
                         marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available."))])
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


@pytest.fixture(
    params=((torch.float64, "cpu", ""),
            (torch.float32, "cpu", ""),
            (torch.float64, "cuda:0", ""),
            (torch.float32, "cuda:0", ""),
            (torch.float64, "cuda:0", "native"),
            (torch.float32, "cuda:0", "native")),
    ids=("cpu64", "cpu32", "cu64", "cu32", "native64", "native32"))
def lattice(request, stencil):
    """Run a test for all lattices (all stencils, devices and data types available on the device.)"""
    dtype, device, native = request.param
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA not available.")
    if device == "cuda:0" and dtype == torch.float32:
        pytest.skip("TODO: loosen tolerances")
    return Lattice(stencil, device=device, dtype=dtype, use_native=(native == "native"))


@pytest.fixture(
    params=((torch.float64, "cpu", "", "cuda:0", ""),
            (torch.float32, "cpu", "", "cuda:0", ""),
            (torch.float64, "cpu", "", "cuda:0", "native"),
            (torch.float32, "cpu", "", "cuda:0", "native"),
            (torch.float64, "cuda:0", "", "cuda:0", "native"),
            (torch.float32, "cuda:0", "", "cuda:0", "native")),
    ids=("cpu_cu_64", "cpu_cu_32", "cpu_native_64", "cpu_native_32", "cu_native_64", "cu_native_32"))
def lattice2(request, stencil):
    """Run a test for all lattices (all stencils, devices and data types available on the device.)"""
    dtype, device, native, device2, native2 = request.param
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA not available.")
    if device == "cuda:0" and dtype == torch.float32:
        pytest.skip("TODO: loosen tolerances")
    if device2 == "cuda:0" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA not available.")
    if device2 == "cuda:0" and dtype == torch.float32:
        pytest.skip("TODO: loosen tolerances")
    return Lattice(stencil, device=device, dtype=dtype, use_native=(native == "native")), \
           Lattice(stencil, device=device2, dtype=dtype, use_native=(native2 == "native"))


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
