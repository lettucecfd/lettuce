from . import *
from . import _load_template, _pretty_print_c


class KernelGenerator:
    cuda: 'NativeCuda' = NativeCuda()
    lattice: 'NativeLattice' = NativeLattice()

    stencil: 'NativeStencil'
    streaming: 'NativeStreaming'
    collision: 'NativeCollision'

    _register: [object] = []

    _launcher_parameter_register: [object]
    _launcher_parameter_name: [str]
    _launcher_parameter_value: [str]
    _launcher_py_parameter_value: [str]

    _kernel_parameter_register: [object]
    _kernel_parameter_name: [str]
    _kernel_parameter_value: [str]

    _constexpr_buffer: [str]
    _index_buffer: [str]
    _node_buffer: [str]
    _distribution_buffer: [str]
    _write_buffer: [str]
    _launcher_buffer: [str]
    _python_wrapper_before_buffer: [str]
    _python_wrapper_after_buffer: [str]

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

        self._register = []

        self._launcher_parameter_register = []
        self._launcher_parameter_name = []
        self._launcher_parameter_value = []
        self._launcher_py_parameter_value = []

        self._kernel_parameter_register = []
        self._kernel_parameter_name = []
        self._kernel_parameter_value = []

        # buffer for pre calculations in different contexts
        self._constexpr_buffer = []
        self._index_buffer = []
        self._node_buffer = []
        self._distribution_buffer = []
        self._write_buffer = []
        self._launcher_buffer = []
        self._python_wrapper_before_buffer = []
        self._python_wrapper_after_buffer = []

        # default parameter
        self.launcher_hook('f', 'at::Tensor f', 'f', 'simulation.f')
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
        return self._register.__contains__(obj)

    def register(self, obj: any):
        self._register.append(obj)

    def launcher_hooked(self, obj: any):
        return self._launcher_parameter_register.__contains__(obj)

    def launcher_hook(self, obj: any, name: str, value: str, py_value: str):
        self._launcher_parameter_register.append(obj)
        self._launcher_parameter_name.append(name)
        self._launcher_parameter_value.append(value)
        self._launcher_py_parameter_value.append(py_value)

    def kernel_hooked(self, obj: any):
        return self._kernel_parameter_register.__contains__(obj)

    def kernel_hook(self, obj: any, name: str, value: str):
        self._kernel_parameter_register.append(obj)
        self._kernel_parameter_name.append(name)
        self._kernel_parameter_value.append(value)

    def append_constexpr_buffer(self, it=''):
        self._constexpr_buffer.append(it)

    def append_launcher_buffer(self, it=''):
        self._launcher_buffer.append(it)

    def append_index_buffer(self, it=''):
        self._index_buffer.append(it)

    def append_node_buffer(self, it=''):
        self._node_buffer.append(it)

    def append_distribution_buffer(self, it=''):
        self._distribution_buffer.append(it)

    def append_write_buffer(self, it=''):
        self._write_buffer.append(it)

    def append_python_wrapper_before_buffer(self, it=''):
        self._python_wrapper_before_buffer.append(it)

    def append_python_wrapper_after_buffer(self, it=''):
        self._python_wrapper_after_buffer.append(it)

    def bake_native(self):
        buffer = _load_template('cuda_kernel').format(
            name=self.name(),
            constexpr_buffer='\n    '.join(self._constexpr_buffer),
            index_buffer='\n    '.join(self._index_buffer),
            node_buffer='\n        '.join(self._node_buffer),
            distribution_buffer='\n            '.join(self._distribution_buffer),
            write_buffer='\n    '.join(self._write_buffer),
            launcher_buffer='\n    '.join(self._launcher_buffer),
            launcher_parameter=', '.join(self._launcher_parameter_name),
            kernel_parameter=', '.join(self._kernel_parameter_name),
            kernel_parameter_values=', '.join(self._kernel_parameter_value))

        if self.pretty_print:
            buffer = _pretty_print_c(buffer)

        return buffer

    def bake_cpp(self):
        buffer = _load_template('cpp_wrapper').format(
            guard=self.header_guard_(),
            name=self.name(),
            launcher_parameter=', '.join(self._launcher_parameter_name),
            launcher_parameter_values=', '.join(self._launcher_parameter_value))

        if self.pretty_print:
            buffer = _pretty_print_c(buffer)

        return buffer

    def bake_py(self, module: str):
        buffer = _load_template('python_wrapper').format(
            name=self.name(),
            python_wrapper_before_buffer='\n    '.join(self._python_wrapper_before_buffer),
            python_wrapper_after_buffer='\n    '.join(self._python_wrapper_after_buffer),
            module=module,
            launcher_py_parameter_value=', '.join(self._launcher_py_parameter_value))

        return buffer
