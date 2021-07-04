#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', "torch>=1.2", "numpy", "matplotlib", "pyevtk"]

setup_requirements = ['pytest-runner', 'pytest']


def get_cmdclass():
    """merge cmdclass of versioneer with the cmdclass of torch's cpp build extension"""

    cmdclass = versioneer.get_cmdclass()

    # This assert should not fail as versioneer (right now) not writes build_ext.
    # This assert should prevent bugs when versioneer changes its behavior.
    assert not ('build_ext' in cmdclass), "versioneer should not write \'build_ext\' in cmdclass." \
                                          "Please contact the developers about this bug!"

    cmdclass['build_ext'] = BuildExtension
    return cmdclass


setup(
    author="Andreas Kraemer",
    author_email='kraemer.research@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Lattice Boltzmann Python GPU",
    entry_points={
        'console_scripts': [
            'lettuce=lettuce.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lettuce',
    name='lettuce',
    packages=find_packages(include=['lettuce', 'lettuce.flows']),
    ext_modules=[
        # CppExtension(
        #    name='lettuce.cpp',
        #    sources=['lettuce/extensions/cpp/lettuce.cpp'],
        #    extra_compile_args=['-fopenmp'],
        #    extra_link_args=['-lgomp']
        # ),
        CUDAExtension(
            name='lettuce._CudaExtension',
            sources=[
                'lettuce/extensions/cuda/lettuce_cuda.cpp',
                'lettuce/extensions/cuda/lettuce_cuda_kernel.cu'
            ]
        )
    ],
    setup_requires=setup_requirements,
    test_suite='tests',
    url='https://github.com/lettucecfd/lettuce',
    version=versioneer.get_version(),
    cmdclass=get_cmdclass(),
    zip_safe=False,
)
