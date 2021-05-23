# PyTorch C++/CUDA extensions, as described in
https://pytorch.org/tutorials/advanced/cpp_extension.html

## Dev Briefing

Create a C++/CUDA extension:
1. Create a C++ source file inside <project-root>/lettuce/cpp/ directory (or a CUDA source file inside <project-root>/lettuce/cuda/ directory)
2. Include "torch/extension.h" and implement the methods you want to expose
3. Create a pybind11 statement to expose the implemented methods
   (Notice that one pybind11 statement matches one ext_module)
4. Add and CppExtension in <project-root>/setup.py

To reuse code write a common header without a pybind11 statement and include it in both modules. (as include and in CppExtension)
