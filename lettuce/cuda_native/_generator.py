import importlib
import os
import os.path
import shutil
import site
import subprocess
import sys
import tempfile
from functools import cached_property
from typing import Dict, Optional, List

from . import *
from .. import __version__, Stencil

__all__ = ['Generator']


class Generator(DefaultCodeGeneration):
    """Formatter & build tool for lattice‐Boltzmann kernels."""

    def __init__(self,
                 stencil: 'Stencil',
                 collision: 'NativeCollision',
                 boundaries: Optional[List['NativeBoundary']] = None,
                 equilibrium: Optional['NativeEquilibrium'] = None,
                 streaming_strategy=StreamingStrategy.POST_STREAMING):

        transformer = [collision] + (boundaries or [])
        DefaultCodeGeneration.__init__(self, stencil, transformer, equilibrium, streaming_strategy)

    @property
    def version(self):
        return __version__

    @cached_property
    def name(self):
        name = [self.stencil.__class__.__name__]
        name += [self.collision.__class__.__name__ if self.collision else 'None']
        name += [self.streaming_strategy.name]
        for b in self.boundaries:
            name += [b.__class__.__name__]
        name += [self.version]
        return lettuce_hash(name)

    def generate(self) -> Dict[str, str]:
        if not hasattr(self, '_generate_result'):
            DefaultCodeGeneration.generate(self)
            buffers = {'name': self.name, 'version': self.version}
            buffers.update(self.joined_buffer())
            setattr(self, '_generate_result', buffers)
        return getattr(self, '_generate_result')

    def format(self, generate_dir: Optional[str] = None) -> str:

        if generate_dir is None:
            generate_dir = tempfile.mkdtemp()

        sources = []

        for input_path_template, input_text_template in template.items():
            val = self.generate()
            input_path = input_path_template.format(**val)
            input_text = input_text_template.format(**val)

            input_dir, input_filename = os.path.split(input_path)

            output_dir: str = os.path.join(generate_dir, input_dir)
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, input_filename)

            if input_path.endswith('.cu') or input_path.endswith('.cpp') or input_path.endswith('.hpp'):
                sources.append(output_path)

            with open(output_path, 'w') as output_file:
                output_file.write(input_text)

        if shutil.which('clang-format') is not None:
            for src in sources:
                subprocess.run(['clang-format', '-i', f"-style=file:{generate_dir}/.clang-format", src], check=True)

        return generate_dir

    def _resolve(self):
        try:
            importlib.reload(site)
            native = importlib.import_module(f"lettuce_{self.name}")
            # noinspection PyUnresolvedReferences
            return native.invoke
        except ModuleNotFoundError:
            print('Could not resolve cuda_native extension.')
            return None
        except AttributeError:
            print('Native module found but it is not working as expected.')
            return None

    def resolve(self, install: bool = True):
        step = self._resolve()

        if step is not None or not install:
            return step

        directory = self.format()
        self.install(directory)

        # do not try to install a second time
        return self.resolve(False)

    @staticmethod
    def install(directory: str):
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
        importlib.reload(site)
