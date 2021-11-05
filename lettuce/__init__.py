# -*- coding: utf-8 -*-

"""Top-level package for lettuce."""

__author__ = 'Andreas Kraemer'
__email__ = 'kraemer.research@gmail.com'

# ==== VERSIONING ====

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# ==== NATIVE CUDA IMPLEMENTATION ====

# stencil is required by native generator to prevent code duplicates
from .stencils import Stencil, D1Q3, D2Q9, D3Q27, D3Q19

import lettuce.native_generator as native_generator

# ==== MODULE IMPORTS ====

from .util import LettuceWarning, InefficientCodeWarning, ExperimentalWarning
from .util import LettuceException, AbstractMethodInvokedError
from .util import all_stencils, get_subclasses
from .util import torch_gradient, torch_jacobi, grid_fine_to_coarse, pressure_poisson
from .unit import UnitConversion

# TODO move below .lattice after equilibrium was moved
from .base import LatticeBase

# TODO equilibrium should not be a member of lattice
#      equilibrium should be a member of collision or independent
from .equilibrium import Equilibrium, QuadraticEquilibrium
from .equilibrium import IncompressibleQuadraticEquilibrium, QuadraticEquilibrium_LessMemory

from .lattices import Lattice
from .force import Guo, ShanChen
from .collision import Collision, NoCollision, BGKCollision, KBCCollision2D, KBCCollision3D, MRTCollision
from .collision import RegularizedCollision, SmagorinskyCollision, TRTCollision, BGKInitialization
from .streaming import Streaming, NoStreaming, StandardStreaming

from .moments import moment_tensor, get_default_moment_transform
from .moments import Moments, Transform, D1Q3Transform, D2Q9Lallemand, D2Q9Dellar, D3Q27Hermite
from .reporters import write_image, write_vtk, VTKReporter, ObservableReporter, ErrorReporter
from .reporters import MaxUReporter, EnergyReporter, EnstrophyReporter, SpectrumReporter
from .boundary import BounceBackBoundary, AntiBounceBackOutlet, EquilibriumBoundaryPU, EquilibriumOutletP
from .observables import Observable, MaximumVelocity, IncompressibleKineticEnergy, Enstrophy, EnergySpectrum

from .simulation import Simulation

# example flows
from .flow import *
