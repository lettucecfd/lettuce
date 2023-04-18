template = {

    "lettuce_native_{name}/__init__.py": """
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
""",

    "lettuce.cpp": """
#include "lettuce.hpp"

#define CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

void
lettuce_{name}({cpp_wrapper_parameter})
{{
    CHECK_CUDA(f); CHECK_CONTIGUOUS(f);
    lettuce_cuda_{name}({cuda_wrapper_parameter_values});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{{
    m.def("collide_and_stream_{name}", &lettuce_{name}, "collide_and_stream_{name}");
}}
""",

    "lettuce.hpp": """
#ifndef {guard}
#define {guard}

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/extension.h>

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

void
lettuce_cuda_{name}({cuda_wrapper_parameter});

void
lettuce_{name}({cpp_wrapper_parameter});

#endif //{guard}
""",

    "lettuce_cuda.cu": """
#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

using index_t = int;
using byte_t = unsigned char;

template<typename scalar_t>
__global__ void
lettuce_cuda_{name}_kernel({kernel_parameter})
{{
  {global_buffer}
  {pipeline_buffer}
}}

void
lettuce_cuda_{name}({cuda_wrapper_parameter})
{{
  {cuda_wrapper_buffer}

  AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "lettuce_cuda_{name}", [&]
  {{
    lettuce_cuda_{name}_kernel<scalar_t><<<block_count, thread_count>>>({kernel_parameter_values});
  }});
}}
""",

    "setup.py": """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path

install_requires = ['torch>=1.2']
setup_requires = []

native_sources = [
    os.path.join(os.path.dirname(__file__), 'lettuce.cpp'),
    os.path.join(os.path.dirname(__file__), 'lettuce_cuda.cu')]

setup(
    author='Robert Andreas Fritsch',
    author_email='info@robert-fritsch.de',
    maintainer='Andreas Kraemer',
    maintainer_email='kraemer.research@gmail.com',
    install_requires=install_requires,
    license='MIT license',
    keywords='lettuce',
    name='lettuce_native_{name}',
    ext_modules=[
        CUDAExtension(
            name='lettuce_native_{name}.native',
            sources=native_sources
        )
    ],
    packages=find_packages(include=['lettuce_native_{name}']),
    setup_requires=setup_requires,
    url='https://github.com/lettucecfd/lettuce',
    version='{version}',
    cmdclass={{'build_ext': BuildExtension}},
    zip_safe=False,
)
""",

}
