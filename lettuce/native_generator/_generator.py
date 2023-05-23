from . import *
from .. import __version__


class Generator:
    """Class Generator

    The generator is one big formatter tool and build tool in one.
    It first uses a configuration of lattice components to generate
    a big dictionary which is then filled into a template directory.
    This directory will then be installed.
    """

    cuda: 'NativeCuda' = NativeCuda()
    lattice: 'NativeLattice' = NativeLattice()
    stencil: 'NativeStencil'
    read: 'NativeRead'
    write: 'NativeWrite'
    pipeline_steps: ['NativePipelineStep']

    reg: {str: [str]}
    par: {str: [str]}
    buf: {str: [str]}

    def __init__(self, stencil, read, write, pipeline_steps):
        self.stencil = stencil
        self.read = read
        self.write = write
        self.pipeline_steps = pipeline_steps
        self.reset()

    def reset(self):
        self.reg = {
            'buffer': [],
            'wrapper_parameter': [],
            'kernel_parameter': []}
        self.par = {
            'cuda_wrapper_parameter_name': [],
            'cuda_wrapper_parameter_value': [],
            'cpp_wrapper_parameter_name': [],
            'cpp_wrapper_parameter_value': [],
            'kernel_parameter_name': [],
            'kernel_parameter_value': []}
        self.buf = {
            'global': [],
            'pipeline': [],
            'cuda_wrapper': [],
            'cpp_wrapper': [],
            'python_wrapper_before': [],
            'python_wrapper_after': []}

    def registered(self, obj: any):
        return self.reg['buffer'].__contains__(obj)

    def register(self, obj: any):
        self.reg['buffer'].append(obj)

    def launcher_hooked(self, obj: any):
        return self.reg['wrapper_parameter'].__contains__(obj)

    def launcher_hook(self, obj: any, name: str, value: str, py_value: str, cond=True):
        if cond:
            self.reg['wrapper_parameter'].append(obj)
            self.par['cuda_wrapper_parameter_name'].append(name)
            self.par['cuda_wrapper_parameter_value'].append(value)
            self.par['cpp_wrapper_parameter_name'].append(name)
            self.par['cpp_wrapper_parameter_value'].append(py_value)

    def kernel_hooked(self, obj: any):
        return self.reg['kernel_parameter'].__contains__(obj)

    def kernel_hook(self, obj: any, name: str, value: str, cond=True):
        if cond:
            self.reg['kernel_parameter'].append(obj)
            self.par['kernel_parameter_name'].append(name)
            self.par['kernel_parameter_value'].append(value)

    def append_global_buffer(self, it='', cond=True):
        if cond:
            self.buf['global'].append(it)

    def append_launcher_buffer(self, it='', cond=True):
        if cond:
            self.buf['cuda_wrapper'].append(it)

    def append_pipeline_buffer(self, it='', cond=True):
        if cond:
            self.buf['pipeline'].append(it)

    def append_python_wrapper_before_buffer(self, it='', cond=True):
        if cond:
            self.buf['python_wrapper_before'].append(it)

    def append_python_wrapper_after_buffer(self, it='', cond=True):
        if cond:
            self.buf['python_wrapper_after'].append(it)

    @property
    def version(self):
        return __version__

    @property
    def name(self):
        # We need to shorten the Name of the Module
        # as it can produce a non-loadable DLL-File otherwise.
        # (We simply hash the Name)

        # python's hash is not reproducible, so it can not be used.
        # murmur3 is the current leading general purpose hash function for hash tables
        # (fast, reproducible, non-cryptographic).
        import mmh3

        collision_name = '_'.join([pipeline_step.name for pipeline_step in self.pipeline_steps])
        name = f"{self.stencil.name}_{self.read.name}_{collision_name}_{self.write.name}_{self.version}"
        return mmh3.hash_bytes(name).hex()

    def generate(self):
        # default parameter
        self.launcher_hook('f', 'at::Tensor f', 'f', 'simulation.f')
        self.kernel_hook('f', 'scalar_t *f', 'f.data<scalar_t>()')

        # default variables
        self.cuda.generate_thread_count(self)
        self.cuda.generate_block_count(self)
        self.stencil.generate_q(self)

        # generate
        self.read.generate(self)
        for pipeline_step in self.pipeline_steps:
            pipeline_step.generate(self)
        self.write.generate(self)

        # convert result
        # noinspection PyDictCreation
        val: {str: str} = {}

        val['name'] = self.name
        val['guard'] = f"LETTUCE_{val['name'].upper()}_HPP"
        val['version'] = self.version

        t1 = '\n    '
        t2 = '\n        '
        t3 = '\n            '
        c1 = ', '

        val['global_buffer'] = t1.join(self.buf['global'])
        val['pipeline_buffer'] = t3.join(self.buf['pipeline'])
        val['cuda_wrapper_buffer'] = t1.join(self.buf['cuda_wrapper'])
        val['cpp_wrapper_buffer'] = t1.join(self.buf['cpp_wrapper'])
        val['python_wrapper_before_buffer'] = t1.join(self.buf['python_wrapper_before'])
        val['python_wrapper_after_buffer'] = t1.join(self.buf['python_wrapper_after'])

        val['kernel_parameter'] = c1.join(self.par['kernel_parameter_name'])
        val['kernel_parameter_values'] = c1.join(self.par['kernel_parameter_value'])

        val['cuda_wrapper_parameter'] = c1.join(self.par['cuda_wrapper_parameter_name'])
        val['cuda_wrapper_parameter_values'] = c1.join(self.par['cuda_wrapper_parameter_value'])

        val['cpp_wrapper_parameter'] = c1.join(self.par['cpp_wrapper_parameter_name'])
        val['cpp_wrapper_parameter_value'] = c1.join(self.par['cpp_wrapper_parameter_value'])

        # release resources
        self.reset()
        return val

    @staticmethod
    def format(val) -> str:
        import os.path
        import tempfile
        from . import template

        temp_dir = tempfile.mkdtemp()

        for input_path_template, input_text_template in template.items():
            input_path = input_path_template.format(**val)
            input_text = input_text_template.format(**val)

            input_dir, input_filename = os.path.split(input_path)

            output_dir = os.path.join(temp_dir, input_dir)
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, input_filename)
            with open(output_path, 'w') as output_file:
                output_file.write(input_text)

        return temp_dir

    def resolve(self):
        try:
            import importlib
            import site
            import re
            importlib.reload(site)
            native = importlib.import_module(f"lettuce_native_{self.name}")
            # noinspection PyUnresolvedReferences
            return native.invoke
        except ModuleNotFoundError:
            print('Could not find the native module. Maybe it is not installed yet.')
            return None
        except AttributeError:
            print('Native module found but it is not working as expected.')
            return None

    @staticmethod
    def install(directory: str):
        import subprocess
        import os
        import sys

        print(f"Installing Native module ({directory}) ...")

        cmd = [sys.executable, 'setup.py', 'install']
        install_log_path = os.path.join(directory, 'install.log')

        with open(install_log_path, 'wb') as install_log:
            p = subprocess.run(cmd, shell=False, cwd=directory, stderr=install_log, stdout=install_log)

        if p.returncode != 0:
            print(f"Install failed! See log-File ({install_log_path}) for more Info.")
            raise subprocess.CalledProcessError(p.returncode, cmd)

        # after install the module is not registered,
        # so we need to reload the register
        import importlib
        import site
        importlib.reload(site)
