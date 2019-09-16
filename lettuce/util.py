"""
Utility functions.
"""

import inspect


class LettuceException(Exception):
    pass


class LettuceWarning(UserWarning):
    pass


class InefficientCodeWarning(LettuceWarning):
    pass


def get_subclasses(classname, module):
    for name, obj in inspect.getmembers(module):
        if hasattr(obj, "__bases__") and classname in obj.__bases__:
            yield obj
