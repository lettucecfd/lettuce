# -*- coding: utf-8 -*-

"""Top-level package for lettuce."""

__author__ = 'Andreas Kraemer'
__email__ = 'kraemer.research@gmail.com'

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

import lettuce.gen_native as gen_native

# import native if available
# else create a pseudo variable
try:
    import lettuce.native as native

    native_available = True

except ImportError:

    class PseudoNative:
        # noinspection PyUnusedLocal
        @staticmethod
        def resolve(*args):
            return None


    native = PseudoNative()
    native_available = False

from .util import LettuceException, LettuceWarning, InefficientCodeWarning, ExperimentalWarning
from .util import get_subclasses, torch_gradient, torch_jacobi, grid_fine_to_coarse, pressure_poisson
from .unit import UnitConversion

# TODO equilibrium should not be a member of lattice
#      equilibrium should be a member of collision
from .equilibrium import Equilibrium, QuadraticEquilibrium
from .equilibrium import IncompressibleQuadraticEquilibrium, QuadraticEquilibrium_LessMemory

from .stencil import Stencil, D1Q3, D2Q9, D3Q27, D3Q19
from .lattice import Lattice
from .force import Guo, ShanChen
from .collision import BGKCollision, KBCCollision2D, KBCCollision3D, MRTCollision
from .collision import RegularizedCollision, SmagorinskyCollision, TRTCollision, BGKInitialization
from .streaming import NoStreaming, StandardStreaming

from .moment import moment_tensor, get_default_moment_transform
from .moment import Moments, Transform, D1Q3Transform, D2Q9Lallemand, D2Q9Dellar, D3Q27Hermite
from .reporter import write_image, write_vtk, VTKReporter, ObservableReporter, ErrorReporter
from .reporter import MaxUReporter, EnergyReporter, EnstrophyReporter, SpectrumReporter
from .boundary import BounceBackBoundary, AntiBounceBackOutlet, EquilibriumBoundaryPU, EquilibriumOutletP
from .observable import Observable, MaximumVelocity, IncompressibleKineticEnergy, Enstrophy, EnergySpectrum

from .simulation import Simulation

# example flows
from .flow import *
