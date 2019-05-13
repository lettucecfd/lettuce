# -*- coding: utf-8 -*-

"""Top-level package for lettuce."""

__author__ = """Andreas Kraemer"""
__email__ = 'kraemer.research@gmail.com'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from lettuce.unit import UnitConversion
from lettuce.lattices import Stencil, D1Q3, D2Q9, Lattice, LatticeOfVector
from lettuce.io import write_png

from lettuce.collision import BGKCollision
from lettuce.streaming import StandardStreaming
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU
from lettuce.io import ErrorReporter
from lettuce.simulation import Simulation

from lettuce.flows import TaylorGreenVortex2D, CouetteFlow2D


