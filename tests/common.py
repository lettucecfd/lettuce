import pytest
import numpy as np
import torch
from copy import copy

from lettuce import *
from lettuce.util.moments import Transform, D1Q3Transform, D2Q9Dellar, \
    D2Q9Lallemand, D3Q27Hermite


def dtype_params():
    return [torch.float64, torch.float32]


def dtype_ids():
    return ['Float64', 'Float32']


def stencil1d_params():
    return [D1Q3()]


def stencil2d_params():
    return [D2Q9()]


def stencil3d_params():
    return [D3Q15(), D3Q19(), D3Q27()]


def stencil_params():
    return stencil1d_params() + stencil2d_params() + stencil3d_params()


def stencil1d_ids():
    return [p.__class__.__name__ for p in stencil1d_params()]


def stencil2d_ids():
    return [p.__class__.__name__ for p in stencil2d_params()]


def stencil3d_ids():
    return [p.__class__.__name__ for p in stencil3d_params()]


def stencil_ids():
    return stencil1d_ids() + stencil2d_ids() + stencil3d_ids()


def device_params():
    return [torch.device('cpu'), torch.device('cuda')]


def device_ids():
    return ['CPU', 'CUDA']


def native_params():
    return [True, False]


def native_ids():
    return ['Native', 'NonNative']


def configuration_params():
    for device in device_params():
        for dtype in dtype_params():
            for native in native_params():
                if not (device == torch.device('cpu') and native):
                    yield device, dtype, native


def configuration_ids():
    buffer = []
    for device in device_ids():
        for dtype in dtype_ids():
            for native in native_ids():
                if not (device == 'CPU' and native == 'Native'):
                    if native == 'Native':
                        buffer.append(f"{device}-{dtype}-{native}")
                    else:
                        buffer.append(f"{device}-{dtype}")
    return buffer


def transform_params():
    Transforms = [
        D1Q3Transform,
        D2Q9Dellar,
        D2Q9Lallemand,
        D3Q27Hermite
    ]
    Stencils = [
        D1Q3,
        D2Q9,
        D2Q9,
        D3Q27
    ]
    return zip(Transforms, Stencils)


def transform_ids():
    return ["D1Q3", "D2Q9Dellar", "D2Q9Lallemand", "D3Q27"]


@pytest.fixture(params=transform_params(), ids=transform_ids())
def fix_transform(request):
    return request.param


COLLISIONS = list(get_subclasses(Collision, lettuce.ext._collision))
@pytest.fixture(params=COLLISIONS)
def fix_collision(request):
    return request.param


def conserving_collision_params():
    return [
        BGKCollision,
        KBCCollision,
        TRTCollision,
        RegularizedCollision,
        SmagorinskyCollision
    ]


def conserving_collision_ids():
    return [
        'BGKCollision',
        'KBCCollision',
        'TRTCollision',
        'RegularizedCollision',
        'SmagorinskyCollision'
    ]


@pytest.fixture(params=dtype_params(), ids=dtype_ids())
def fix_dtype(request):
    return request.param


@pytest.fixture(params=stencil1d_params(), ids=stencil1d_ids())
def fix_stencil1d(request):
    return request.param


@pytest.fixture(params=stencil2d_params(), ids=stencil2d_ids())
def fix_stencil2d(request):
    return request.param


@pytest.fixture(params=stencil3d_params(), ids=stencil3d_ids())
def fix_stencil3d(request):
    return request.param


@pytest.fixture(params=stencil_params(), ids=stencil_ids())
def fix_stencil(request):
    return request.param


@pytest.fixture(params=device_params(), ids=device_ids())
def fix_device(request):
    if 'cuda' in request.param and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available on this machine.",
                    allow_module_level=True)
    return request.param


@pytest.fixture(params=native_params(), ids=native_ids())
def fix_native(request):
    if request.param[0] and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available on this machine.",
                    allow_module_level=True)
    return request.param


@pytest.fixture(params=configuration_params(), ids=configuration_ids())
def fix_configuration(request):
    if 'cuda' in request.param[0].type and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available on this machine.",
                    allow_module_level=True)
    return request.param


@pytest.fixture(params=conserving_collision_params(),
                ids=conserving_collision_ids())
def fix_conserving_collision(request):
    return request.param


class TestFlow(ExtFlow):
    __test__ = False

    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None):
        self._boundaries = []
        super().__init__(context, resolution, reynolds_number, mach_number,
                         stencil, equilibrium)

    def make_resolution(self, resolution: List[int],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            if stencil is None:
                return [resolution]
            else:
                return [resolution] * stencil.d
        else:
            return resolution

    def make_units(self, reynolds_number, mach_number,
                   resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(reynolds_number, mach_number,
                              characteristic_length_lu=resolution[0])

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        u = 1.01 * np.ones([self.stencil.d] + self.resolution)
        p = 0.01 * np.ones([1] + self.resolution)
        return p, u

    @property
    def boundaries(self) -> List['Boundary']:
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries: List['Boundary']):
        self._boundaries = boundaries


def DummyTGV(context: 'Context', resolution: Union[int, List[int]],
             reynolds_number, mach_number,
             stencil: Optional['Stencil'] = None,
             equilibrium: Optional['Equilibrium'] = None):
    return TaylorGreenVortex(context, resolution, reynolds_number, mach_number,
                             stencil, equilibrium)


class DummyFlow(ExtFlow):

    def __init__(self, context: Context, resolution: int = 16):
        ExtFlow.__init__(self, context, resolution, 1.0, 1.0)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        return [resolution, resolution] if isinstance(resolution, int)\
            else resolution

    def make_units(self, reynolds_number, mach_number, _: List[int]
                   ) -> 'UnitConversion':
        return UnitConversion(reynolds_number=reynolds_number,
                              mach_number=mach_number)

    def initial_pu(self) -> (float, List[float]):
        ...

    def initialize(self):
        ...

    @property
    def boundaries(self) -> List['Boundary']:
        return []
