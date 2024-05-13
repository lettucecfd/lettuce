import json
from typing import Optional, List, AnyStr

from . import *
from .. import __version__, Stencil

__all__ = [
    'Generator',
]


class Generator:
    """Class Generator

    The generator is one big formatter tool and build tool in one.
    It first uses a configuration of lattice components to generate
    a big dictionary which is then filled into a template directory.
    This directory will then be installed.
    """

    reg: {AnyStr: [AnyStr]}
    par: {AnyStr: [AnyStr]}
    buf: {AnyStr: [AnyStr]}

    # noinspection PyDefaultArgument
    def __init__(self, stencil: 'Stencil', collision: Optional['NativeCollision'] = None, boundaries: List['NativeBoundary'] = [],
                 equilibrium: Optional['NativeEquilibrium'] = None):

        self.stencil = stencil
        self.collision = collision
        self.boundaries = boundaries
        self.equilibrium = equilibrium

        if len(self.boundaries) == 0:
            self.support_no_collision_mask = False
            self.support_no_streaming_mask = False
        else:
            self.support_no_collision_mask = True
            self.support_no_streaming_mask = True

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

    def launcher_hook(self, obj: any, name: AnyStr, value: AnyStr, py_value: AnyStr, cond=True):
        if cond:
            self.reg['wrapper_parameter'].append(obj)
            self.par['cuda_wrapper_parameter_name'].append(name)
            self.par['cuda_wrapper_parameter_value'].append(value)
            self.par['cpp_wrapper_parameter_name'].append(name)
            self.par['cpp_wrapper_parameter_value'].append(py_value)

    def kernel_hooked(self, obj: any):
        return self.reg['kernel_parameter'].__contains__(obj)

    def kernel_hook(self, obj: any, name: AnyStr, value: AnyStr, cond=True):
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
        name = self.stencil.__class__.__name__
        name += '_' + self.collision.__class__.__name__
        for boundary in self.boundaries:
            name += '_' + boundary.__class__.__name__
        name += '_' + self.version
        return mmh3.hash_bytes(name).hex()

    def generate(self):
        # noinspection PyDictCreation
        val: {AnyStr: AnyStr} = {}

        d = self.stencil.d
        q = self.stencil.q

        val['debug'] = 0
        val['support_no_collision_mask'] = int(self.support_no_collision_mask)
        val['support_no_streaming_mask'] = int(self.support_no_streaming_mask)

        val['d'] = str(d)
        val['q'] = str(q)
        val['cs'] = json.dumps(self.stencil.cs)

        val['e'] = '{'
        for it in self.stencil.e:
            val['e'] += '{' + ', '.join([str(j) for j in it]) + '},'
        val['e'] += '}'

        val['w'] = '{'
        for w in self.stencil.w:
            val['w'] += f"{json.dumps(w)},"
        val['w'] += '}'

        # generate
        self.collision.generate(self)
        for boundary in self.boundaries:
            boundary.generate(self)

        # convert result

        val['name'] = self.name
        val['version'] = self.version

        t1 = '\n    '
        t3 = '\n            '
        c1 = ', '

        val['global_buffer'] = t1.join(self.buf['global'])
        val['pipeline_buffer'] = t3.join(self.buf['pipeline'])
        val['cuda_wrapper_buffer'] = t1.join(self.buf['cuda_wrapper'])
        val['cpp_wrapper_buffer'] = t1.join(self.buf['cpp_wrapper'])
        val['python_wrapper_before_buffer'] = t1.join(self.buf['python_wrapper_before'])
        val['python_wrapper_after_buffer'] = t1.join(self.buf['python_wrapper_after'])

        val['kernel_parameter'] = ''.join([c1 + p for p in self.par['kernel_parameter_name']])
        val['kernel_parameter_values'] = ''.join([c1 + p for p in self.par['kernel_parameter_value']])

        val['cuda_wrapper_parameter'] = ''.join([c1 + p for p in self.par['cuda_wrapper_parameter_name']])
        val['cuda_wrapper_parameter_values'] = ''.join([c1 + p for p in self.par['cuda_wrapper_parameter_value']])

        val['cpp_wrapper_parameter'] = ''.join([c1 + p for p in self.par['cpp_wrapper_parameter_name']])
        val['cpp_wrapper_parameter_value'] = ''.join([c1 + p for p in self.par['cpp_wrapper_parameter_value']])

        # release resources
        self.reset()
        return val

    @staticmethod
    def format(val) -> AnyStr:
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

    def _resolve(self):
        try:
            import importlib
            import site
            import re
            importlib.reload(site)
            native = importlib.import_module(f"lettuce_native_{self.name}")
            # noinspection PyUnresolvedReferences
            return native.invoke
        except ModuleNotFoundError:
            print('Could not resolve native extension.')
            return None
        except AttributeError:
            print('Native module found but it is not working as expected.')
            return None

    def resolve(self, install=True):
        step = self._resolve()

        if step is not None or not install:
            return step

        buffer = self.generate()
        directory = self.format(buffer)
        self.install(directory)

        # do not try to install a second time
        return self.resolve(False)

    @staticmethod
    def install(directory: AnyStr):
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
