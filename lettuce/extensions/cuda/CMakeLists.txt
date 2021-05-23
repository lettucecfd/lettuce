# Warning!!! This File is not used to actually build the Extension!
# The purpose of this File is simply to allow Editors to understand the code!
cmake_minimum_required(VERSION 3.1)

set(THIS CudaExt)
project(${THIS} LANGUAGES CUDA)

# Fetch all the Information necessary to build the Cuda Extension

file(GLOB CUDA_CU_SOURCES *.cu)
file(GLOB CUDA_CUH_SOURCES *.cuh)

execute_process(COMMAND python -c "from torch.utils.cpp_extension import _find_cuda_home; print(_find_cuda_home(), end='')" OUTPUT_VARIABLE CUDA_HOME_DIR)
set(CUDA_INCLUDE_DIR ${CUDA_HOME_DIR}/include)

execute_process(COMMAND python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc(), end='')" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
execute_process(COMMAND python -c "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()), end='')" OUTPUT_VARIABLE PYTORCH_INCLUDE_DIRS)
execute_process(COMMAND python -c "from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths()), end='')" OUTPUT_VARIABLE PYTORCH_LINK_DIRS)

# Setup the Library Recipe

add_library(${THIS})
set_target_properties(${THIS} PROPERTIES
        LINKER_LANGUAGE            CUDA
        CUDA_STANDARD              11
        CUDA_STANDARD_REQUIRED     ON
        CUDA_SEPARABLE_COMPILATION ON)

foreach(CUDA_SOURCE IN LISTS CUDA_CU_SOURCES CUDA_CUH_SOURCES)
    target_sources(${THIS} PUBLIC ${CUDA_SOURCE})
    message(STATUS "Added Source  File      to Target ${THIS}: ${CUDA_SOURCE}")
endforeach()

foreach(INCLUDE_DIRECTORY IN LISTS CUDA_INCLUDE_DIR PYTHON_INCLUDE_DIR PYTORCH_INCLUDE_DIRS)
    #string(REPLACE "\n" "" INCLUDE_DIRECTORY ${INCLUDE_DIRECTORY})
    target_include_directories(${THIS} PUBLIC ${INCLUDE_DIRECTORY})
    message(STATUS "Added Include Directory to Target ${THIS}: ${INCLUDE_DIRECTORY}")
endforeach()

foreach(LINK_DIRECTORY IN LISTS PYTORCH_LINK_DIRS)
    #string(REPLACE "\n" "" LINK_DIRECTORY ${LINK_DIRECTORY})
    target_link_directories(${THIS} PUBLIC ${LINK_DIRECTORY})
    message(STATUS "Added Link    Directory to Target ${THIS}: ${LINK_DIRECTORY}")
endforeach()
