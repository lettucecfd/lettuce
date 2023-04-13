#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path

install_requires = ['torch>=1.2']
setup_requires = []

native_sources = [
    os.path.join(os.path.dirname(__file__), 'lettuce.cpp'),
    os.path.join(os.path.dirname(__file__), 'lettuce_cuda.cu')]

setup(
    author='Robert Andreas Fritsch',
    author_email='info@robert-fritsch.de',
    maintainer='Andreas Kraemer',
    maintainer_email='kraemer.research@gmail.com',
    install_requires=install_requires,
    license='MIT license',
    keywords='lettuce',
    name='lettuce_native_{name}',
    ext_modules=[
        CUDAExtension(
            name='lettuce_native_{name}.native',
            sources=native_sources
        )
    ],
    packages=find_packages(include=['lettuce_native_{name}']),
    setup_requires=setup_requires,
    url='https://github.com/lettucecfd/lettuce',
    version='{version}',
    cmdclass={{'build_ext': BuildExtension}},
    zip_safe=False,
)
