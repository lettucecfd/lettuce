"""Top-level package for lettuce."""

from importlib.metadata import (PackageNotFoundError as _PackageNotFoundError,
                                metadata as _metadata)

try:
    # Distribution name, keep in sync with [project] name in pyproject.toml.
    # Everything below is read from the installed metadata so that it cannot
    # drift away from pyproject.toml and AUTHORS.rst.
    _dist = _metadata('lettucecfd')
except _PackageNotFoundError:
    # Running from a source tree without the package being installed.
    __version__ = '0.0.0+unknown'
    __author__ = ''
    __email__ = ''
else:
    __version__ = _dist['Version']
    __author__ = _dist['Author'] or ''
    __email__ = _dist['Maintainer-email'] or ''

from ._context import *
from ._stencil import *
from ._unit import *

from ._flow import *
from ._simulation import *

import lettuce.util
import lettuce.ext

from lettuce.util import *
from lettuce.ext import *
