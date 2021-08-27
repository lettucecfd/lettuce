# -*- coding: utf-8 -*-

"""Top-level package for lettuce."""

__author__ = """Andreas Kraemer"""
__email__ = 'kraemer.research@gmail.com'


# ==== VERSIONING ====

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


# ==== NATIVE CUDA IMPLEMENTATION ====


import lettuce.gen_native as gen_native

# import native if available
# else create a pseudo variable
try:
    import lettuce.native as native

    native_available = True

except ImportError:
    print('failed to load native module')
    native_available = False


    class PseudoNative:
        # noinspection PyUnusedLocal
        @staticmethod
        def resolve(*args):
            return None


    native = PseudoNative()


# ==== MODULE IMPORTS ====

from lettuce.util import *
from lettuce.unit import *
from lettuce.lattices import *
from lettuce.equilibrium import *
from lettuce.stencils import *
from lettuce.moments import *
from lettuce.reporters import *

from lettuce.collision import *
from lettuce.streaming import *
from lettuce.boundary import *
from lettuce.reporters import *
from lettuce.simulation import *
from lettuce.force import *
from lettuce.observables import *

from lettuce.flows import *
