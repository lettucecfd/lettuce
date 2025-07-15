from typing import Dict

__all__ = [
    'template',
]

template: Dict[str, str] = {
".clang-format": """
BasedOnStyle: Mozilla
TabWidth: 2
UseTab: Never
ColumnLimit: 0
MaxEmptyLinesToKeep: 1
AllowShortFunctionsOnASingleLine: Empty
AllowShortLambdasOnASingleLine: Empty
AlignConsecutiveAssignments: true
""",

    "lettuce_{name}/__init__.py": """
import torch
import numpy as np

import os
import importlib

__all__ = ['cuda_native', 'invoke']

if os.name == 'nt':
    # on windows add cuda path for cuda_native module to find all dll's
    os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

cuda_native = importlib.import_module("lettuce_{name}.cuda_native")


def invoke(simulation):
    {python_pre}
    cuda_native.lettuce({py_values})
    {python_post}
    simulation.flow.f, simulation.flow.f_next = simulation.flow.f_next, simulation.flow.f
""",

    "lettuce.cpp": """
#include "lettuce.hpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){{m.def("lettuce",&lettuce,"lettuce");}}
""",

    "lettuce.hpp": """
#ifndef LETTUCE_HPP
#define LETTUCE_HPP
#include <torch/extension.h>
void lettuce({cuda_parameters});
#endif
""",

    "lettuce_cuda.cu": """
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

using index_t = int;
using byte_t = unsigned char;

template<typename scalar_t>
__global__ void
lettuce_kernel({kernel_parameters})
{{
  {pipes}
  {pipe}
}}

void
lettuce({cuda_parameters})
{{
  {cuda}
  AT_DISPATCH_FLOATING_TYPES(f.scalar_type(),"lettuce",[&]{{
    lettuce_kernel<scalar_t><<<block_count, thread_count>>>(
      {cuda_values});
  }});
  torch::cuda::synchronize(-1);
}}
""",

    "setup.py": """
#!/usr/bin/env python

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join, dirname

native_sources = [
    join(dirname(__file__), 'lettuce.cpp'),
    join(dirname(__file__), 'lettuce_cuda.cu')]

extra_compile_args = {{
    'cxx': [],
    'nvcc': [
        '--ptxas-options',
        '-v,--warn-on-spills,--allow-expensive-optimizations=true,-O3',
        '--use_fast_math',
        '--optimize', '3',
        '--maxrregcount', '128'
    ]
}}

setup(
    author='Robert Andreas Fritsch',
    author_email='info@robert-fritsch.de',
    maintainer='Robert Andreas Fritsch',
    maintainer_email='info@robert-fritsch.de',
    install_requires=['torch>=1.2'],
    license='MIT license',
    keywords='lettuce',
    name='lettuce_{name}',
    ext_modules=[
        CUDAExtension(
            name='lettuce_{name}.cuda_native',
            sources=native_sources,
            extra_compile_args=extra_compile_args
        )
    ],
    packages=find_packages(include=['lettuce_{name}']),
    setup_requires=[],
    url='https://github.com/lettucecfd/lettuce',
    version='{version}',
    cmdclass={{'build_ext': BuildExtension}},
    zip_safe=False,
)
""",

}
