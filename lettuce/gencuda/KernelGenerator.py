import re

from lettuce.gencuda import *

cuda_frame = '''
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

using index_t = unsigned int;
using byte_t = unsigned char;

template<typename scalar_t>
__global__ void
lettuce_cuda_{signature}_kernel({kernel_parameter})
{{
    {hard_buffer}

    {index_buffer}
    
        {node_buffer}

#pragma unroll
        for (index_t i = 0; i < q; ++i)
        {{
            {collision_buffer}
        }}
        
    {write_buffer}
}}

void
lettuce_cuda_{signature}({wrapper_parameter})
{{
    {wrapper_buffer}

    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "lettuce_cuda_{signature}", [&]
    {{
        lettuce_cuda_{signature}_kernel<scalar_t><<<block_count, thread_count>>>({kernel_parameter_values});
        cudaDeviceSynchronize();
    }});
}}
'''

cpp_frame = '''
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
lettuce_cuda_{signature}({wrapper_parameter});

#define CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

void
{signature}({wrapper_parameter})
{{
    CHECK_CUDA(f);
    CHECK_CONTIGUOUS(f);

    lettuce_cuda_{signature}({wrapper_parameter_values});
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{{
    m.def("{signature}", &{signature}, "{signature}");
}}
'''

py_frame = '''
def {signature}(simulation: 'lettuce.Simulation'):
    """"""
    {py_buffer}
    # noinspection PyUnresolvedReferences
    {module}.{signature}({py_parameter_values})
'''


class KernelGenerator:
    register_: [object] = []

    wrapper_parameter_register_: [object]
    wrapper_parameter_signature_: [str]
    wrapper_parameter_value_: [str]
    wrapper_py_parameter_value_: [str]

    kernel_parameter_register_: [object]
    kernel_parameter_signature_: [str]
    kernel_parameter_value_: [str]

    hard_buffer_: [str]
    index_buffer_: [str]
    node_buffer_: [str]
    collision_buffer_: [str]
    write_buffer_: [str]
    wrapper_buffer_: [str]
    py_buffer_: [str]

    signature_: str

    def __init__(self,
                 stencil: 'Stencil',
                 stream: 'Stream',
                 equilibrium: 'Equilibrium',
                 collision: 'Collision',

                 support_no_stream: bool,
                 support_no_collision: bool,

                 cuda: 'Cuda' = None,
                 lattice: 'Lattice' = None,

                 pretty_print: bool = False):
        """
        :param cuda:
        :param lattice:
        :param stencil:
        :param stream:
        :param equilibrium:
        :param collision:
        :param support_no_stream:
        :param support_no_collision:
        :param pretty_print:
        """

        self.stencil = stencil
        self.stream = stream
        self.equilibrium = equilibrium
        self.collision = collision

        self.cuda = cuda if cuda is not None else Cuda()
        self.lattice = lattice if lattice is not None else Lattice()

        self.pretty_print = pretty_print

        #
        # reset
        #

        self.register_ = []

        self.wrapper_parameter_register_ = []
        self.wrapper_parameter_signature_ = []
        self.wrapper_parameter_value_ = []
        self.wrapper_py_parameter_value_ = []

        self.kernel_parameter_register_ = []
        self.kernel_parameter_signature_ = []
        self.kernel_parameter_value_ = []

        # buffer for pre calculations in different contexts
        self.hard_buffer_ = []
        self.index_buffer_ = []
        self.node_buffer_ = []
        self.collision_buffer_ = []
        self.write_buffer_ = []
        self.wrapper_buffer_ = []
        self.py_buffer_ = []

        self.signature_ = ''

        # default parameter
        self.wrapper_hook('f', 'at::Tensor f', 'f', 'simulation.f')
        self.kernel_hook('f', 'scalar_t *f', 'f.data<scalar_t>()')

        # default variables
        self.cuda.thread_count(self)
        self.cuda.block_count(self)
        self.stencil.q(self)

        #
        # generate
        #

        extra = ''
        if support_no_stream:
            extra += '_nsm'
        if support_no_collision:
            extra += '_ncm'

        self.signature_ = f"{self.stencil.name}_{self.equilibrium.name}_{self.collision.name}_{self.stream.name}{extra}"
        self.stream.read_write(self, support_no_stream, support_no_collision)
        self.collision.collide(self)

    def signature(self):
        return self.signature_

    def cuda_file_name(self):
        return f"lettuce_cuda_{self.signature()}.cu"

    def cpp_file_name(self):
        return f"lettuce_cuda_{self.signature()}.cpp"

    def registered(self, obj: any):
        return self.register_.__contains__(obj)

    def register(self, obj: any):
        self.register_.append(obj)

    def wrapper_hooked(self, obj: any):
        return self.wrapper_parameter_register_.__contains__(obj)

    def wrapper_hook(self, obj: any, signature: str, value: str, py_value: str):
        self.wrapper_parameter_register_.append(obj)
        self.wrapper_parameter_signature_.append(signature)
        self.wrapper_parameter_value_.append(value)
        self.wrapper_py_parameter_value_.append(py_value)

    def kernel_hooked(self, obj: any):
        return self.kernel_parameter_register_.__contains__(obj)

    def kernel_hook(self, obj: any, signature: str, value: str):
        self.kernel_parameter_register_.append(obj)
        self.kernel_parameter_signature_.append(signature)
        self.kernel_parameter_value_.append(value)

    def hrd(self, it=''):
        self.hard_buffer_.append(it)

    def wrp(self, it=''):
        self.wrapper_buffer_.append(it)

    def idx(self, it=''):
        self.index_buffer_.append(it)

    def nde(self, it=''):
        self.node_buffer_.append(it)

    def cln(self, it=''):
        self.collision_buffer_.append(it)

    def wrt(self, it=''):
        self.write_buffer_.append(it)

    def py(self, it=''):
        self.py_buffer_.append(it)

    def pretty_print_c_(self, buffer: str):
        if self.pretty_print:
            # remove spaces before new line
            buffer = re.sub(r' +\n', '\n', buffer)
            # remove multiple empty lines
            buffer = re.sub(r'\n\n\n+', '\n\n', buffer)
            # remove whitespace at end and begin
            buffer = re.sub(r'\n+$', '\n', buffer)
            buffer = re.sub(r'^\n*', '', buffer)
            # place preprocessor directives at start of line
            buffer = re.sub(r'\n +#', '\n#', buffer)
            # remove unnecessary whitespace between closures
            buffer = re.sub(r'{\n\n+', '{\n', buffer)
            buffer = re.sub(r'}\n\n+(\s*)}', r'}\n\1}', buffer)
        return buffer

    def bake_cuda(self):
        buffer = cuda_frame.format(signature=self.signature(),
                                   hard_buffer='\n    '.join(self.hard_buffer_),
                                   index_buffer='\n    '.join(self.index_buffer_),
                                   node_buffer='\n        '.join(self.node_buffer_),
                                   collision_buffer='\n            '.join(self.collision_buffer_),
                                   write_buffer='\n    '.join(self.write_buffer_),
                                   wrapper_buffer='\n    '.join(self.wrapper_buffer_),
                                   wrapper_parameter=', '.join(self.wrapper_parameter_signature_),
                                   kernel_parameter=', '.join(self.kernel_parameter_signature_),
                                   kernel_parameter_values=', '.join(self.kernel_parameter_value_))

        return self.pretty_print_c_(buffer)

    def bake_cpp(self):
        buffer = cpp_frame.format(signature=self.signature(),
                                  wrapper_parameter=', '.join(self.wrapper_parameter_signature_),
                                  wrapper_parameter_values=', '.join(self.wrapper_parameter_value_))

        return self.pretty_print_c_(buffer)

    def bake_py(self, module: str):
        buffer = py_frame.format(signature=self.signature(),
                                 py_buffer='\n    '.join(self.py_buffer_),
                                 module=module,
                                 py_parameter_values=', '.join(self.wrapper_py_parameter_value_))

        return buffer
