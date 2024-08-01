"""
Fixtures for unit tests.
"""
import pytest

import numpy as np
import torch

from lettuce import *
from lettuce.ext import _stencil

STENCILS = list(get_subclasses(Stencil, _stencil))


@pytest.fixture(
    params=((torch.float64, "cpu", "no_native"),
            (torch.float32, "cpu", "no_native"),
            (torch.float64, "cuda", "no_native"),
            (torch.float32, "cuda", "no_native"),
            (torch.float64, "cuda", "cuda_native"),
            (torch.float32, "cuda", "cuda_native")),
    ids=("cpu64", "cpu32", "cu64", "cu32", "native64", "native32"))
def configurations(request):
    dtype, device, native = request.param
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA not available.")
    return dtype, device, native


@pytest.fixture(params=STENCILS)
def stencils(request):
    """Run a test for all _stencil."""
    return request.param


class TestFlow(ExtFlow):

    def make_resolution(self, resolution: List[int]) -> List[int]:
        if isinstance(resolution, int):
            return [resolution]
        else:
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(reynolds_number, mach_number, characteristic_length_lu=resolution[0])

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        u = 1.0 * np.ones([self.stencil.d] + self.resolution)
        p = 0.0 * np.ones([1] + self.resolution)
        return p, u
