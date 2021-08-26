import os
import shutil

from lettuce.gen_native import *
from lettuce.gen_native.generator_util import pretty_print_c_, pretty_print_py_

py_frame = '''
import torch
import os

# on windows add cuda path for
# native module to find all dll's
if os.name == 'nt':
    os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import {module}

{py_buffer}

def resolve(stencil: str, equilibrium: str, collision: str, stream: str = None,
            support_no_stream: bool = None, support_no_collision: bool = None):
    """"""

    stream = stream if stream is not None else "standard"

    extra = ''
    if support_no_stream is not None and support_no_stream:
        extra += '_nsm'
    if support_no_collision is not None and support_no_collision:
        extra += '_ncm'

    signature = f"{{stencil}}_{{equilibrium}}_{{collision}}_{{stream}}{{extra}}"
    if signature in globals():
        return globals()[signature]

    return None
'''

pybind_frame = '''
#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/extension.h>

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

{includes}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{{
    {definitions}
}}
'''


class GeneratorModule:
    """
    """

    class Matrix:
        """
        """

        stencil: ['NativeStencil']
        stream: ['NativeStreaming']
        equilibrium: ['NativeEquilibrium']
        collision: ['NativeCollision']

        # by default all of these
        # versions should be created
        support_no_stream: [bool]
        support_no_collision: [bool]

        def __init__(self,
                     stencil: ['NativeStencil'],
                     stream: ['NativeStreaming'],
                     equilibrium: ['NativeEquilibrium'],
                     collision: ['NativeCollision'],
                     support_no_stream: [bool] = None,
                     support_no_collision: [bool] = None):
            """
            """

            self.stencil = stencil
            self.stream = stream
            self.equilibrium = equilibrium
            self.collision = collision

            if support_no_stream is not None:
                self.support_no_stream = support_no_stream
            else:
                self.support_no_stream = [True, False]

            if support_no_collision is not None:
                self.support_no_collision = support_no_collision
            else:
                self.support_no_collision = [True, False]

        def entries_(self):
            matrix_entries = []
            for a0 in self.stencil:
                for a1 in self.stream:
                    for a2 in self.equilibrium:
                        for a3 in self.collision:
                            for a4 in self.support_no_stream:
                                for a5 in self.support_no_collision:
                                    matrix_entries.append((a0, a1, a2, a3, a4, a5))
            return matrix_entries

        @staticmethod
        def gen_from_entry_(entry, pretty_print: bool):
            return GeneratorKernel(stencil=entry[0],
                                   streaming=entry[1],
                                   equilibrium=entry[2],
                                   collision=entry[3],
                                   support_no_stream=entry[4],
                                   support_no_collision=entry[5],
                                   pretty_print=pretty_print)

    matrix_list: []
    pretty_print: bool

    def __init__(self,
                 white_matrices: ['GeneratorModule.Matrix'],
                 black_matrices: ['GeneratorModule.Matrix'] = None,
                 pretty_print: bool = False):
        """
        :param white_matrices:
        :param black_matrices:
        :param pretty_print:
        """

        self.matrix_list = []
        for white_matrix in white_matrices:
            self.matrix_list += white_matrix.entries_()
        if black_matrices is not None:
            for black_matrix in black_matrices:
                self.matrix_list -= black_matrix.entries_()

        self.matrix_list = set(self.matrix_list)

        self.pretty_print = pretty_print

    def create_module(self):

        py_module = os.path.join('lettuce', 'native')
        native_module = 'lettuce_native'

        shutil.rmtree(native_module, ignore_errors=True)
        os.mkdir(native_module)
        shutil.rmtree(py_module, ignore_errors=True)
        os.mkdir(py_module)

        buffer: [str] = []
        pybind_include_buffer: [str] = []
        pybind_definition_buffer: [str] = []

        for entry in self.matrix_list:
            gen = GeneratorModule.Matrix.gen_from_entry_(entry, self.pretty_print)

            cuda_file = open(os.path.join(native_module, gen.native_file_name()), 'w')
            cuda_file.write(gen.bake_native())
            cuda_file.close()

            cpp_file = open(os.path.join(native_module, gen.cpp_file_name()), 'w')
            cpp_file.write(gen.bake_cpp())
            cpp_file.close()

            buffer.append(gen.bake_py(native_module))

            pybind_include_buffer.append(f'#include "{gen.cpp_file_name()}"')
            pybind_definition_buffer.append(f'm.def("{gen.signature()}", '
                                            f'&{gen.signature()}, '
                                            f'"{gen.signature()}");')

        buffer = py_frame.format(module=native_module, py_buffer='\n'.join(buffer))
        if self.pretty_print:
            buffer = pretty_print_py_(buffer)
        module_file = open(os.path.join(py_module, '__init__.py'), 'w')
        module_file.write(buffer)
        module_file.close()

        buffer = pybind_frame.format(includes='\n'.join(pybind_include_buffer),
                                     definitions='\n    '.join(pybind_definition_buffer))
        if self.pretty_print:
            buffer = pretty_print_c_(buffer)
        module_file = open(os.path.join(native_module, 'lettuce_pybind.cpp'), 'w')
        module_file.write(buffer)
        module_file.close()
