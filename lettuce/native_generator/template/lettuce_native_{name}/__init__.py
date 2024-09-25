# noinspection PyUnresolvedReferences
import torch


def _import_lettuce_native():
    import importlib
    return importlib.import_module("lettuce_native_{name}.native")


def _ensure_cuda_path():
    import os

    # on windows add cuda path for
    # native module to find all dll's
    if os.name == 'nt':
        os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))


# do not expose the os and importlib package
_ensure_cuda_path()
native = _import_lettuce_native()

# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyStatementEffect
def collide_and_stream(simulation):
    {python_wrapper_before_buffer}
    native.collide_and_stream_{name}({cpp_wrapper_parameter_value})
    torch.cuda.synchronize()
    {python_wrapper_after_buffer}
