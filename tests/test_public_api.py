"""
Keep the public namespace and the API reference in docs/modules.rst in sync.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import lettuce

MODULES_RST = Path(__file__).parents[1] / "docs" / "modules.rst"

# The modules whose names `lettuce/__init__.py` re-exports with `import *`.
EXPORTING_MODULES = [
    "lettuce._context",
    "lettuce._stencil",
    "lettuce._unit",
    "lettuce._flow",
    "lettuce._simulation",
    "lettuce.util",
] + [f"lettuce.ext.{m.name}"
     for m in pkgutil.iter_modules(lettuce.ext.__path__)]


def public_names():
    names = set()
    for module in EXPORTING_MODULES:
        names |= set(importlib.import_module(module).__all__)
    return names


def documented_names():
    text = MODULES_RST.read_text()
    return set(re.findall(
        r"^\.\. (?:autoclass|autofunction|autoexception|autodata|data)"
        r":: (\w+)", text, flags=re.MULTILINE))


@pytest.mark.parametrize("module", EXPORTING_MODULES)
def test_exporting_module_defines_all(module):
    # Without __all__, `import *` also re-exports the module's own imports
    # (numpy, torch, typing helpers, ...) into the lettuce namespace.
    assert hasattr(importlib.import_module(module), "__all__")


def test_public_names_are_importable():
    missing = sorted(n for n in public_names() if not hasattr(lettuce, n))
    assert not missing


def test_namespace_contains_no_foreign_names():
    foreign = sorted(
        name for name, obj in vars(lettuce).items()
        if not name.startswith("_")
        and not inspect.ismodule(obj)
        and name not in public_names())
    assert not foreign


@pytest.mark.skipif(not MODULES_RST.exists(),
                    reason="docs are not part of this checkout")
def test_every_public_name_is_documented():
    undocumented = sorted(public_names() - documented_names())
    assert not undocumented, \
        f"add these to docs/modules.rst: {undocumented}"


@pytest.mark.skipif(not MODULES_RST.exists(),
                    reason="docs are not part of this checkout")
def test_every_documented_name_exists():
    stale = sorted(n for n in documented_names() if not hasattr(lettuce, n))
    assert not stale, f"remove these from docs/modules.rst: {stale}"
