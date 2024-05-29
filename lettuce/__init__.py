"""Top-level package for lettuce."""

__author__ = 'Andreas Kraemer'
__email__ = 'kraemer.research@gmail.com'

import lettuce._version

__version__ = _version.get_versions()['version']

from ._context import *
from ._stencil import *
from ._unit import *

from ._flow import *
from ._simulation import *

import lettuce.util
import lettuce.ext

from lettuce.util import *
from lettuce.ext import *
