from . import *
from .generator_util import _pretty_print_c

native_frame = '''
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
    {{
        {node_buffer}

#pragma unroll
        for (index_t i = 0; i < q; ++i)
        {{
            {collision_buffer}
        }}
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
    }});
}}
'''

cpp_frame = '''
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

#endif //{guard}
'''

py_frame = '''
def {signature}(simulation):
    """"""
    {py_pre_buffer}
    # noinspection PyUnresolvedReferences
    {module}.{signature}({py_parameter_values})
    torch.cuda.synchronize()
    {py_post_buffer}
'''


class GeneratorKernel:
    cuda: 'NativeCuda' = NativeCuda()
    lattice: 'NativeLattice' = NativeLattice()

    stencil: 'NativeStencil'
    streaming: 'NativeStreaming'
    collision: 'NativeCollision'

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
    py_pre_buffer_: [str]
    py_post_buffer_: [str]

    signature_: str

    def __init__(self,
                 stencil: 'NativeStencil',
                 streaming: 'NativeStreaming',
                 collision: 'NativeCollision',
                 pretty_print: bool = False):

        self.stencil = stencil
        self.streaming = streaming
        self.collision = collision
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
        self.py_pre_buffer_ = []
        self.py_post_buffer_ = []

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

        self.signature_ = f"{self.stencil.name}_{self.collision.name}_{self.streaming.name}"

        self.streaming.read_write(self)
        self.collision.collision(self)

    def signature(self):
        return self.signature_

    def header_guard_(self):
        return f"lettuce_{self.signature()}_hpp".upper()

    def native_file_name(self):
        return f"lettuce_cuda_{self.signature()}.cu"

    def cpp_file_name(self):
        return f"lettuce_{self.signature()}.hpp"

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

    def pyr(self, it=''):
        self.py_pre_buffer_.append(it)

    def pyo(self, it=''):
        self.py_post_buffer_.append(it)

    def bake_native(self):
        buffer = native_frame.format(
            signature=self.signature(),
            hard_buffer='\n    '.join(self.hard_buffer_),
            index_buffer='\n    '.join(self.index_buffer_),
            node_buffer='\n        '.join(self.node_buffer_),
            collision_buffer='\n            '.join(self.collision_buffer_),
            write_buffer='\n    '.join(self.write_buffer_),
            wrapper_buffer='\n    '.join(self.wrapper_buffer_),
            wrapper_parameter=', '.join(self.wrapper_parameter_signature_),
            kernel_parameter=', '.join(self.kernel_parameter_signature_),
            kernel_parameter_values=', '.join(self.kernel_parameter_value_))

        if self.pretty_print:
            buffer = _pretty_print_c(buffer)

        return buffer

    def bake_cpp(self):
        buffer = cpp_frame.format(
            guard=self.header_guard_(),
            signature=self.signature(),
            wrapper_parameter=', '.join(self.wrapper_parameter_signature_),
            wrapper_parameter_values=', '.join(self.wrapper_parameter_value_))

        if self.pretty_print:
            buffer = _pretty_print_c(buffer)

        return buffer

    def bake_py(self, module: str):
        buffer = py_frame.format(
            signature=self.signature(),
            py_pre_buffer='\n    '.join(self.py_pre_buffer_),
            py_post_buffer='\n    '.join(self.py_post_buffer_),
            module=module,
            py_parameter_values=', '.join(self.wrapper_py_parameter_value_))

        return buffer
