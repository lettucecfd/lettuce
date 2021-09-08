from . import *
from . import _load_template, _pretty_print_c


class KernelGenerator:
    cuda: 'NativeCuda' = NativeCuda()
    lattice: 'NativeLattice' = NativeLattice()

    stencil: 'NativeStencil'
    streaming: 'NativeStreaming'
    collision: 'NativeCollision'

    register_: [object] = []

    wrapper_parameter_register_: [object]
    wrapper_parameter_name_: [str]
    wrapper_parameter_value_: [str]
    wrapper_py_parameter_value_: [str]

    kernel_parameter_register_: [object]
    kernel_parameter_name_: [str]
    kernel_parameter_value_: [str]

    hard_buffer_: [str]
    index_buffer_: [str]
    node_buffer_: [str]
    collision_buffer_: [str]
    write_buffer_: [str]
    wrapper_buffer_: [str]
    py_pre_buffer_: [str]
    py_post_buffer_: [str]

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
        self.wrapper_parameter_name_ = []
        self.wrapper_parameter_value_ = []
        self.wrapper_py_parameter_value_ = []

        self.kernel_parameter_register_ = []
        self.kernel_parameter_name_ = []
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

        # default parameter
        self.wrapper_hook('f', 'at::Tensor f', 'f', 'simulation.f')
        self.kernel_hook('f', 'scalar_t *f', 'f.data<scalar_t>()')

        # default variables
        self.cuda.generate_thread_count(self)
        self.cuda.generate_block_count(self)
        self.stencil.generate_q(self)

        #
        # generate
        #

        self.streaming.generate_read_write(self)
        self.collision.generate_collision(self)

    def name(self):
        return f"{self.stencil.name}_{self.collision.name}_{self.streaming.name}"

    def header_guard_(self):
        return f"lettuce_{self.name()}_hpp".upper()

    def native_file_name(self):
        return f"lettuce_cuda_{self.name()}.cu"

    def cpp_file_name(self):
        return f"lettuce_{self.name()}.hpp"

    def registered(self, obj: any):
        return self.register_.__contains__(obj)

    def register(self, obj: any):
        self.register_.append(obj)

    def wrapper_hooked(self, obj: any):
        return self.wrapper_parameter_register_.__contains__(obj)

    def wrapper_hook(self, obj: any, name: str, value: str, py_value: str):
        self.wrapper_parameter_register_.append(obj)
        self.wrapper_parameter_name_.append(name)
        self.wrapper_parameter_value_.append(value)
        self.wrapper_py_parameter_value_.append(py_value)

    def kernel_hooked(self, obj: any):
        return self.kernel_parameter_register_.__contains__(obj)

    def kernel_hook(self, obj: any, name: str, value: str):
        self.kernel_parameter_register_.append(obj)
        self.kernel_parameter_name_.append(name)
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
        buffer = _load_template('cuda_kernel').format(
            name=self.name(),
            hard_buffer='\n    '.join(self.hard_buffer_),
            index_buffer='\n    '.join(self.index_buffer_),
            node_buffer='\n        '.join(self.node_buffer_),
            collision_buffer='\n            '.join(self.collision_buffer_),
            write_buffer='\n    '.join(self.write_buffer_),
            wrapper_buffer='\n    '.join(self.wrapper_buffer_),
            wrapper_parameter=', '.join(self.wrapper_parameter_name_),
            kernel_parameter=', '.join(self.kernel_parameter_name_),
            kernel_parameter_values=', '.join(self.kernel_parameter_value_))

        if self.pretty_print:
            buffer = _pretty_print_c(buffer)

        return buffer

    def bake_cpp(self):
        buffer = _load_template('cpp_wrapper').format(
            guard=self.header_guard_(),
            name=self.name(),
            wrapper_parameter=', '.join(self.wrapper_parameter_name_),
            wrapper_parameter_values=', '.join(self.wrapper_parameter_value_))

        if self.pretty_print:
            buffer = _pretty_print_c(buffer)

        return buffer

    def bake_py(self, module: str):
        buffer = _load_template('python_wrapper').format(
            name=self.name(),
            py_pre_buffer='\n    '.join(self.py_pre_buffer_),
            py_post_buffer='\n    '.join(self.py_post_buffer_),
            module=module,
            py_parameter_values=', '.join(self.wrapper_py_parameter_value_))

        return buffer
